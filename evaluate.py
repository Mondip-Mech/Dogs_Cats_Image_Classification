"""evaluate.py — Generate evaluation metrics from the trained model.

Computes accuracy, precision, recall, F1, AUC-ROC, confusion matrix, and
per-class classification report, then saves everything to
metrics/evaluation_results.json so the Streamlit app can display them.

Usage
-----
    python evaluate.py --val-dir data/validation

Expected directory structure for --val-dir:
    validation/
        cats/   (or Cat/)
        dogs/   (or Dog/)

The script auto-detects the subfolder names.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import CLASS_NAMES, IMG_SIZE, METRICS_PATH, MODEL_PATH


def find_class_dirs(val_dir: Path) -> dict[str, int]:
    """Map subfolder names to class indices (alphabetical order matches Keras default)."""
    subdirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    if len(subdirs) != 2:
        raise ValueError(f"Expected exactly 2 class subdirectories, found {len(subdirs)}: {subdirs}")
    return {d.name: i for i, d in enumerate(subdirs)}


def build_generator(val_dir: Path, class_map: dict[str, int]):
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    return datagen.flow_from_directory(
        str(val_dir),
        target_size=IMG_SIZE,
        batch_size=64,
        class_mode="binary",
        shuffle=False,
        classes={k: v for k, v in class_map.items()},
    )


def evaluate(val_dir: Path) -> dict:
    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    class_map = find_class_dirs(val_dir)
    print(f"Class mapping: {class_map}")

    generator = build_generator(val_dir, class_map)
    print(f"Running inference on {generator.n} images…")

    y_prob = model.predict(generator, verbose=1).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    y_true = generator.classes

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = float(roc_auc_score(y_true, y_prob))

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )

    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro")),
        "recall":    float(recall_score(y_true, y_pred, average="macro")),
        "f1":        float(f1_score(y_true, y_pred, average="macro")),
        "auc_roc":   auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": report,
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": auc,
        },
        "model_comparison": [
            {
                "Model": "Custom CNN (3 Conv layers)",
                "Parameters": "6.45 M",
                "Val Accuracy": "78.4%",
                "Val Loss": "0.459",
                "Epochs to best": 10,
            },
            {
                "Model": "MobileNetV2 — frozen base",
                "Parameters": "2.42 M",
                "Val Accuracy": "96.90%",
                "Val Loss": "0.0745",
                "Epochs to best": 8,
            },
            {
                "Model": "MobileNetV2 — fine-tuned (last 30 layers)",
                "Parameters": "2.42 M",
                "Val Accuracy": "97.08%",
                "Val Loss": "0.0708",
                "Epochs to best": 19,
            },
        ],
    }


def print_summary(results: dict) -> None:
    print("\n" + "=" * 45)
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1 Score  : {results['f1']:.4f}")
    print(f"  AUC-ROC   : {results['auc_roc']:.4f}")
    print("=" * 45)
    cm = results["confusion_matrix"]
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"           Cat    Dog")
    print(f"  Cat   {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"  Dog   {cm[1][0]:5d}  {cm[1][1]:5d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Dogs vs Cats model.")
    parser.add_argument(
        "--val-dir",
        type=Path,
        required=True,
        help="Path to validation directory containing two class subfolders.",
    )
    args = parser.parse_args()

    if not args.val_dir.exists():
        print(f"Error: validation directory not found: {args.val_dir}", file=sys.stderr)
        sys.exit(1)

    results = evaluate(args.val_dir)
    print_summary(results)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nMetrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
