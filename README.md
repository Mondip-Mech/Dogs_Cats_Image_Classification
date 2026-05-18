# Dogs vs Cats Image Classification

**End-to-end deep learning pipeline** — from exploratory data analysis through transfer learning, fine-tuning, and production deployment — achieving **97.08% validation accuracy** on 25,000 images.

---

## Overview

This project demonstrates a complete machine learning workflow for binary image classification:

1. **EDA** — dataset validation, class balance check, sample visualisation
2. **Baseline modelling** — custom CNN to establish a performance floor
3. **Transfer learning** — MobileNetV2 pretrained on ImageNet, head only trained
4. **Fine-tuning** — last 30 base layers unfrozen at very low learning rate
5. **Deployment** — interactive Streamlit app with GradCAM explainability

---

## Results

| Model | Parameters | Val Accuracy | Val Loss | Notes |
|---|---|---|---|---|
| Custom CNN (3 conv layers) | 6.45 M | 78.4% | 0.459 | Underfits — too shallow |
| MobileNetV2 (frozen base) | 2.42 M | 96.90% | 0.0745 | Big jump from ImageNet features |
| **MobileNetV2 (fine-tuned)** | **2.42 M** | **97.08%** | **0.0708** | **Production model** |

### Evaluation Metrics (5,000 validation images)

| Metric | Cat | Dog | Macro Avg |
|---|---|---|---|
| Precision | 97.42% | 96.74% | 97.08% |
| Recall | 96.72% | 97.44% | 97.08% |
| F1 Score | 97.07% | 97.09% | 97.08% |
| AUC-ROC | — | — | **0.9968** |

---

## Model Architecture

```
Input (128 × 128 × 3)
    │
    ▼
MobileNetV2 (ImageNet pretrained)
  └─ Last 30 layers unfrozen for fine-tuning
  └─ Output: (4 × 4 × 1280)
    │
    ▼
GlobalAveragePooling2D  →  (1280,)
    │
    ▼
Dropout (rate = 0.3)
    │
    ▼
Dense (128, ReLU)
    │
    ▼
Dense (1, Sigmoid)  →  P(Dog)
```

**Total parameters:** 2,422,081  
**Trainable (fine-tune phase):** 164,097 (head) + last 30 base layers

---

## Three-Phase Training Strategy

### Phase 1 — Baseline CNN
Custom 3-layer CNN trained from scratch. Reached 78.4% validation accuracy after 10 epochs. Established a performance baseline and confirmed that a shallow network underfits this dataset.

### Phase 2 — Transfer Learning (frozen base)
MobileNetV2 base fully frozen, only the custom head trained. Validation accuracy jumped to **96.90%** in just 8 epochs, confirming that ImageNet visual features transfer well to pet images.

### Phase 3 — Fine-tuning (last 30 layers unfrozen)
Base layers from index `[-30:]` unfrozen and trained at `lr = 1e-5` with `ReduceLROnPlateau`. Validation accuracy reached **97.08%** at epoch 19. Fine-tuning adapted higher-level features (textures, shapes) to the pet domain.

---

## Explainability — GradCAM

The Streamlit app includes a **Gradient-weighted Class Activation Map (GradCAM)** visualisation. This highlights which regions of the input image most influenced the model's decision, providing interpretability beyond a raw confidence score.

Warm colours (red/yellow) indicate high model attention; cool colours (blue) indicate low attention.

---

## Data Augmentation Pipeline

| Technique | Parameter | Purpose |
|---|---|---|
| Rescaling | ÷ 255 | Normalise to [0, 1] |
| Rotation | ± 40° | Orientation invariance |
| Horizontal flip | 50% | Left/right symmetry |
| Zoom | ± 20% | Scale invariance |
| Shear | ± 20% | Perspective robustness |

---

## Project Structure

```
Dogs_Cats_Image_Classification/
│
├── app.py                          # Streamlit dashboard (4 tabs)
├── config.py                       # Centralised configuration
├── evaluate.py                     # Standalone evaluation script
├── requirements.txt
├── README.md
│
├── metrics/
│   └── evaluation_results.json     # Pre-computed metrics (confusion matrix, ROC, etc.)
│
└── Dogs_Cats_Image_Classification.ipynb   # Training notebook
```

---

## How to Run

### Option A — Local (conda)

```bash
conda create -n dogs_cats python=3.10
conda activate dogs_cats
pip install -r requirements.txt
python -m streamlit run app.py
```

### Option B — Docker

```bash
# Build (requires best_model.keras in the project root)
docker build -t dogs-cats-classifier .

# Run
docker run -p 8501:8501 dogs-cats-classifier
```

The app opens at `http://localhost:8501` with four tabs:

- **Single Prediction** — upload an image, get prediction + GradCAM heatmap
- **Batch Prediction** — upload multiple images, download results as CSV
- **Evaluation Dashboard** — confusion matrix, ROC curve, classification report
- **Architecture** — model layers, training config, data augmentation details

### Regenerate evaluation metrics

If you have access to the validation data:

```bash
python evaluate.py --val-dir path/to/validation
```

This recomputes all metrics from the saved model and overwrites `metrics/evaluation_results.json`.

---

## Experiment Tracking — MLflow

All three training phases are logged as separate MLflow runs for side-by-side comparison.

```bash
pip install mlflow
# Run the MLflow cells in the notebook, then:
mlflow ui
# Open: http://localhost:5000
```

The MLflow UI shows parameter diffs and metric comparisons across Custom CNN → Frozen MobileNetV2 → Fine-tuned MobileNetV2 in a single table.

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep learning | TensorFlow 2.15 · Keras |
| Base model | MobileNetV2 (ImageNet) |
| Data | Kaggle Dogs vs Cats (25,000 images) |
| Evaluation | scikit-learn (metrics, ROC) |
| Visualisation | Matplotlib · Streamlit |
| Explainability | GradCAM (custom implementation) |
| Deployment | Streamlit |

---

## Dataset

[Kaggle Dogs vs Cats](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset) — 24,998 images after validation (12,499 cats · 12,499 dogs).

**Split:** 80% train (19,998) · 20% validation (5,000)
