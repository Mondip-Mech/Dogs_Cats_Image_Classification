"""Dogs vs Cats Image Classifier — Advanced Streamlit Dashboard.

Features:
  - Single prediction with GradCAM explainability heatmap
  - Batch prediction with downloadable CSV
  - Evaluation dashboard (confusion matrix, ROC curve, classification report)
  - Model architecture & training strategy overview
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from config import MODEL_PATH, METRICS_PATH, IMG_SIZE, CLASS_NAMES, GRADCAM_LAYER, CONFIDENCE_THRESHOLD

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dogs vs Cats Classifier",
    page_icon="🐾",
    layout="wide",
)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))


@st.cache_resource
def load_validator():
    """MobileNetV2 on ImageNet — used to check if image is actually a cat/dog."""
    return tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


# ImageNet class indices that correspond to cats and dogs
_DOG_CLASS_RANGE = set(range(151, 269))   # 151–268: all dog breeds
_CAT_CLASSES     = {281, 282, 283, 284, 285}  # tabby, tiger cat, Persian, Siamese, Egyptian
_PET_CLASSES     = _DOG_CLASS_RANGE | _CAT_CLASSES


def is_pet_image(img: Image.Image, validator) -> bool:
    """Return True only if ImageNet top-5 predictions include at least one cat/dog class."""
    arr = np.array(img.resize((224, 224))).astype(np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    preds = validator.predict(np.expand_dims(arr, 0), verbose=0)
    top5 = np.argsort(preds[0])[::-1][:5]
    return bool(set(top5) & _PET_CLASSES)


# ── Inference helpers ─────────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    arr = np.array(img.resize(IMG_SIZE)) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)


def run_inference(model, img: Image.Image):
    arr = preprocess(img)
    raw = float(model.predict(arr, verbose=0)[0][0])
    label = CLASS_NAMES[1] if raw > 0.5 else CLASS_NAMES[0]
    confidence = raw if raw > 0.5 else 1.0 - raw
    return label, confidence, arr


# ── GradCAM ───────────────────────────────────────────────────────────────────
def make_gradcam_heatmap(img_array: np.ndarray, model) -> np.ndarray | None:
    """Compute GradCAM heatmap targeting the last spatial feature map.

    Searches top-level layers first, then dives into any nested sub-model
    (e.g. MobileNetV2 base) to find the last Conv layer with 4D output.
    """
    last_spatial = None

    # Build a flat list: top-level layers, then inner layers of any sub-model
    candidates = list(reversed(model.layers))
    for layer in model.layers:
        if hasattr(layer, "layers"):          # it's a nested model
            candidates.extend(reversed(layer.layers))

    for layer in candidates:
        if layer.__class__.__name__ == "InputLayer":
            continue
        try:
            shape = layer.output.shape
            if len(shape) == 4:
                last_spatial = layer
                break
        except AttributeError:
            continue

    if last_spatial is None:
        return None

    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_spatial.output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            class_signal = preds[:, 0]

        grads = tape.gradient(class_signal, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(tf.nn.relu(heatmap))
        max_val = tf.reduce_max(heatmap)
        if float(max_val) == 0:
            return None
        return (heatmap / max_val).numpy()
    except Exception:
        return None


def overlay_heatmap(img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize(img.size, Image.BILINEAR)
    )
    colormap = plt.cm.jet(heatmap_resized)[:, :, :3]
    colored = np.uint8(colormap * 255)
    blended = np.uint8(colored * alpha + np.array(img) * (1 - alpha))
    return Image.fromarray(blended)


# ── Layout ────────────────────────────────────────────────────────────────────
st.title("🐾 Dogs vs Cats Classifier")
st.caption(
    "MobileNetV2 Transfer Learning  ·  97.08% Validation Accuracy  ·  GradCAM Explainability"
)

model = load_model()
validator = load_validator()
metrics = load_metrics()

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single Prediction",
    "📦 Batch Prediction",
    "📊 Evaluation Dashboard",
    "🧠 Architecture",
])


# ── Tab 1: Single Prediction ──────────────────────────────────────────────────
with tab1:
    st.subheader("Classify a single image")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="single")

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        label, confidence, arr = run_inference(model, img)
        heatmap = make_gradcam_heatmap(arr, model)

        # Two-step validation: ImageNet check takes priority over confidence score
        pet_detected = is_pet_image(img, validator)
        low_confidence = confidence < CONFIDENCE_THRESHOLD
        is_ood = not pet_detected or low_confidence   # out-of-distribution

        col_img, col_cam, col_result = st.columns(3)

        with col_img:
            st.image(img, caption="Input Image", use_container_width=True)

        with col_cam:
            if is_ood:
                st.info("GradCAM is only shown for confirmed cat/dog images.")
            elif heatmap is not None:
                overlay = overlay_heatmap(img, heatmap)
                st.image(overlay, caption="GradCAM — Model Attention Map", use_container_width=True)
                st.caption(
                    "Warm colours show regions the model focused on when making its decision."
                )
            else:
                st.info("GradCAM visualisation not available for this model configuration.")

        with col_result:
            if is_ood:
                reason = "not recognised as a cat or dog by ImageNet" if not pet_detected else f"low confidence ({confidence * 100:.1f}%)"
                st.warning(
                    f"**This does not appear to be a cat or dog.**\n\n"
                    f"Reason: {reason}.\n\n"
                    "Please upload a clear photo of a cat or dog."
                )
                st.caption(
                    f"The binary classifier guessed **{label}** ({confidence * 100:.1f}%) "
                    "but this result is not reliable for non-pet images."
                )
            else:
                st.metric("Prediction", label)
                st.metric("Confidence", f"{confidence * 100:.2f}%")
                st.progress(float(confidence))

                dog_p = confidence if label == "Dog" else 1.0 - confidence
                cat_p = 1.0 - dog_p
                prob_df = pd.DataFrame(
                    {"Probability": [round(cat_p, 4), round(dog_p, 4)]},
                    index=CLASS_NAMES,
                )
                st.bar_chart(prob_df)


# ── Tab 2: Batch Prediction ───────────────────────────────────────────────────
with tab2:
    st.subheader("Classify multiple images at once")
    files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch",
    )

    if files:
        rows = []
        bar = st.progress(0, text="Running inference…")
        for i, f in enumerate(files):
            img = Image.open(f).convert("RGB")
            label, conf, _ = run_inference(model, img)
            dog_p = conf if label == "Dog" else 1.0 - conf
            rows.append({
                "Filename": f.name,
                "Prediction": label if conf >= CONFIDENCE_THRESHOLD else "⚠️ Uncertain",
                "Confidence (%)": round(conf * 100, 2),
                "Dog Probability (%)": round(dog_p * 100, 2),
                "Cat Probability (%)": round((1 - dog_p) * 100, 2),
                "Flag": "Low confidence" if conf < CONFIDENCE_THRESHOLD else "",
            })
            bar.progress((i + 1) / len(files), text=f"Processed {i + 1}/{len(files)}")

        df = pd.DataFrame(rows)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Images", len(df))
        m2.metric("Dogs", int((df["Prediction"] == "Dog").sum()))
        m3.metric("Cats", int((df["Prediction"] == "Cat").sum()))
        m4.metric("Uncertain", int((df["Prediction"] == "⚠️ Uncertain").sum()))

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )


# ── Tab 3: Evaluation Dashboard ───────────────────────────────────────────────
with tab3:
    st.subheader("Validation Set Evaluation  (5,000 images · 2,500 cats · 2,500 dogs)")

    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy",  f"{metrics['accuracy']:.2%}")
        c2.metric("Precision", f"{metrics['precision']:.2%}")
        c3.metric("Recall",    f"{metrics['recall']:.2%}")
        c4.metric("F1 Score",  f"{metrics['f1']:.2%}")
        c5.metric("AUC-ROC",   f"{metrics['auc_roc']:.4f}")

        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.markdown("**Confusion Matrix**")
            cm = np.array(metrics["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(4, 3))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(CLASS_NAMES)
            ax.set_yticklabels(CLASS_NAMES)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            for i in range(2):
                for j in range(2):
                    color = "white" if cm[i, j] > cm.max() / 2 else "black"
                    ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                            color=color, fontsize=13, fontweight="bold")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_roc:
            st.markdown("**ROC Curve**")
            fpr = metrics["roc"]["fpr"]
            tpr = metrics["roc"]["tpr"]
            auc_val = metrics["roc"]["auc"]
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(fpr, tpr, color="#1565C0", lw=2, label=f"MobileNetV2  AUC = {auc_val:.4f}")
            ax.fill_between(fpr, tpr, alpha=0.08, color="#1565C0")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("**Per-Class Classification Report**")
        report = metrics["classification_report"]
        display_keys = [k for k in report if k != "accuracy"]
        report_df = pd.DataFrame({k: report[k] for k in display_keys}).T
        st.dataframe(
            report_df.style.format("{:.4f}", na_rep="—"),
            use_container_width=True,
        )

        st.markdown("**Model Comparison: Three-Phase Training**")
        st.dataframe(
            pd.DataFrame(metrics["model_comparison"]),
            use_container_width=True,
        )

    else:
        st.warning(
            "No evaluation metrics found.  \n"
            "Run `python evaluate.py --val-dir data/validation` to generate them, "
            "then restart the app."
        )


# ── Tab 4: Architecture ───────────────────────────────────────────────────────
with tab4:
    st.subheader("Model Architecture & Training Strategy")

    col_arch, col_train = st.columns(2)

    with col_arch:
        st.markdown("**Architecture**")
        st.table(pd.DataFrame({
            "Layer": [
                "Input",
                "MobileNetV2 base",
                "GlobalAveragePooling2D",
                "Dropout",
                "Dense (hidden)",
                "Dense (output)",
            ],
            "Details": [
                "128 × 128 × 3",
                "ImageNet pretrained — last 30 layers unfrozen",
                "1280-dim feature vector",
                "rate = 0.3",
                "128 units, ReLU",
                "1 unit, Sigmoid → binary probability",
            ],
            "Parameters": ["—", "2,257,984", "0", "0", "163,968", "129"],
        }))

    with col_train:
        st.markdown("**Training Configuration**")
        st.table(pd.DataFrame({
            "Parameter": [
                "Input size", "Batch size", "Loss function",
                "Phase 1 optimizer", "Phase 2 LR (fine-tune)",
                "Callbacks",
            ],
            "Value": [
                "128 × 128 px", "512", "Binary Crossentropy",
                "Adam (default LR)", "1e-6 → 1e-5 (ReduceLROnPlateau)",
                "EarlyStopping · ReduceLROnPlateau · ModelCheckpoint",
            ],
        }))

    st.markdown("**Three-Phase Training Strategy**")
    st.table(pd.DataFrame({
        "Phase": ["1 — Baseline CNN", "2 — Transfer Learning", "3 — Fine-tuning"],
        "Approach": [
            "Custom 3-layer CNN trained from scratch",
            "MobileNetV2 base fully frozen, only head trained",
            "Last 30 base layers unfrozen, very low LR (1e-5)",
        ],
        "Val Accuracy": ["78.4%", "96.90%", "97.08%"],
        "Val Loss":     ["0.459",  "0.0745", "0.0708"],
        "Key Insight": [
            "Too shallow for 25k images — underfits",
            "ImageNet features transfer well to pet photos — big accuracy jump",
            "Adapting higher-level features to the domain squeezes residual error",
        ],
    }))

    st.markdown("**Data Augmentation Pipeline**")
    st.table(pd.DataFrame({
        "Technique": ["Rescaling", "Rotation", "Horizontal Flip", "Zoom", "Shear"],
        "Parameter":  ["÷ 255", "± 40°", "50% probability", "± 20%", "± 20%"],
        "Purpose": [
            "Normalise pixel values to [0, 1]",
            "Orientation invariance",
            "Left/right symmetry",
            "Scale invariance",
            "Perspective robustness",
        ],
    }))
