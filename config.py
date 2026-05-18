from pathlib import Path

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "best_model.keras"
METRICS_PATH = BASE_DIR / "metrics" / "evaluation_results.json"

IMG_SIZE = (128, 128)
CLASS_NAMES = ["Cat", "Dog"]

# Predictions below this confidence are flagged as out-of-distribution
CONFIDENCE_THRESHOLD = 0.90

# Name of the last spatial layer in the loaded model (MobileNetV2 base block).
# Used by GradCAM to extract feature maps. Update if you change the base model.
GRADCAM_LAYER = "mobilenetv2_1.00_128"

TRAIN_CONFIG = {
    "batch_size": 512,
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "phase1_epochs": 18,        # frozen base
    "phase2_epochs": 20,        # fine-tune last 30 layers
    "finetune_lr": 1e-5,
    "dropout_rate": 0.3,
    "fine_tuned_layers": 30,
    "augmentation": {
        "rotation_range": 40,
        "shear_range": 0.2,
        "zoom_range": 0.2,
        "horizontal_flip": True,
        "fill_mode": "nearest",
    },
}
