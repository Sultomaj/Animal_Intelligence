# Fine-Grained Animal Species Classification via Transfer Learning and Stacked Ensemble Architecture

**Course Project · Deep Learning · 2025**

---

## Abstract

This work presents a complete end-to-end pipeline for 90-class animal species classification using convolutional neural networks trained under a transfer learning paradigm. Three architectures were evaluated — MobileNetV2, EfficientNetB0, and a custom stacked ensemble — on a curated 90-class image dataset. EfficientNetB0 achieved the highest validation accuracy of **87.04%**, while a post-hoc analysis of the ensemble's underperformance revealed a non-obvious PyTorch `BatchNorm` behaviour that degraded the frozen feature extractors during joint training. A publicly accessible inference application is deployed on Hugging Face Spaces.

---

## 1. Problem Statement

Automated fine-grained visual recognition of animal species is a well-studied but practically demanding task. The intra-class appearance variation (e.g., juvenile vs. adult morphology, seasonal colouration) and inter-class visual similarity (e.g., deer vs. reindeer, rat vs. mouse) make 90-way classification a non-trivial benchmark. This project treats the task as a controlled study in comparing lightweight and mid-capacity backbone architectures under a frozen-feature fine-tuning regime.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | Kaggle — *Animal Image Dataset: 90 Different Animals* |
| Total images | ~5,400 |
| Classes | 90 |
| Train / Val split | 80% / 20% (stratified random) |

---

## 3. Methodology

### 3.1 Preprocessing & Augmentation

A deliberately asymmetric augmentation strategy was applied: the training pipeline introduces stochastic perturbations while the validation pipeline applies only deterministic resizing. This is the intended cause of the observed phenomenon where validation accuracy consistently exceeds training accuracy in early epochs — not data leakage, but augmentation difficulty.

**Training transforms:** `RandomResizedCrop(224)` · `RandomHorizontalFlip` · `RandomRotation(±20°)` · `ColorJitter(b=0.2, c=0.2, s=0.2, h=0.2)` · ImageNet normalisation

**Validation transforms:** `Resize(224)` · `CenterCrop(224)` · ImageNet normalisation

### 3.2 Architectures

| Model | Backbone | Trainable Parameters | Strategy |
|---|---|---|---|
| MobileNetV2 | MobileNetV2 (ImageNet) | Classifier head only | Frozen backbone, weighted CE loss |
| EfficientNetB0 | EfficientNetB0 (ImageNet) | Classifier head only | Frozen backbone, weighted CE loss |
| StackedEnsemble | MobileNetV2 + EfficientNetB0 | Meta-classifier `Linear(180 → 90)` | Both bases frozen |

Class-frequency-weighted cross-entropy was used for the two base models to account for mild class imbalance. The ensemble used unweighted CE since the meta-classifier operates over logit concatenations, not raw pixel distributions.

**Optimiser:** Adam · **LR:** 1e-3 · **Scheduler:** CosineAnnealingLR (T_max = 10) · **Epochs:** 10 · **Batch size:** 32

---

## 4. Results

### 4.1 Quantitative Summary

| Model | Val Accuracy | Val Loss |
|---|---|---|
| MobileNetV2 | 86.48% | 0.567 |
| EfficientNetB0 | **87.04%** | 0.678 |
| StackedEnsemble | 81.02% | 0.796 |

### 4.2 Learning Curves

Both base models exhibit a healthy convergence pattern: validation accuracy rises steeply in the first three epochs and plateaus around epoch 5–6, with training accuracy converging from below. The ~15% train–val accuracy gap persists at convergence and is attributable to augmentation difficulty rather than underfitting — the frozen backbone never receives augmented gradients, so the classifier head is optimised against a harder distribution than it is evaluated on.

The ensemble converges more slowly (starting near chance at epoch 1) and finishes with a wider, reversed gap (train ~65%, val ~81%), indicative of the `BatchNorm` corruption described in Section 4.4.

### 4.3 Per-Class Analysis

The confusion matrices are well-diagonalised across all models. Persistent failure modes are concentrated in three clusters:

**Visually ambiguous ungulates:** donkey (40–50%), ox (46–53%), deer (46–69% depending on model). These classes share body plan and colouration with horse, cow, bison, and reindeer respectively.

**Small mammals with overlapping texture:** hamster (31–69%), possum (54%), rat (47–68%). Confusion is concentrated between rodent and marsupial classes.

**Lepidoptera and related insects:** moth performs worst across all models (9–64%), frequently misclassified as butterfly. The dataset likely contains low-resolution or heavily cropped specimens where wing-pattern discriminability is lost.

EfficientNetB0 recovers hamster (+38 pp over MobileNetV2) and antelope (+12 pp) but regresses on duck (−14 pp) and goat (−36 pp). No single model dominates across all classes.

### 4.4 Technical Finding: BatchNorm Corruption in Frozen Ensemble Training

The ensemble's 6-point accuracy deficit relative to its base models is explained by an interaction between PyTorch's `model.train()` call and `BatchNorm2d` running statistics.

When `StackedEnsemble.train()` is called during the meta-classifier training loop, PyTorch recursively propagates training mode to all child modules, including the frozen MobileNetV2 and EfficientNetB0 backbones. While `requires_grad=False` correctly prevents weight updates, it does **not** suppress `BatchNorm2d`'s update of its running mean and variance buffers. These buffers are not parameters in the autograd sense — they are registered as buffers and update unconditionally in training mode.

The effect is that the running statistics, originally calibrated on ImageNet-scale feature distributions, are progressively corrupted by the smaller, task-specific batch statistics of the ensemble training loader. This injects noise into the backbone's feature representations before they reach the meta-classifier, degrading both base models simultaneously.

**Mitigation:** Explicitly call `base_model.eval()` on each frozen sub-network after constructing the ensemble and before the training loop. This freezes the `BatchNorm` running statistics while allowing the meta-classifier to train normally.

```python
ensemble.model1.eval()
ensemble.model2.eval()
# Then proceed with the training loop — only ensemble.classifier trains
```

---

## 5. Deployment

The MobileNetV2 model is deployed as an interactive Streamlit application on Hugging Face Spaces.

**Application features:**
- Real-time inference with confidence-threshold warnings (threshold: 60%)
- Contextual species information via Wikipedia API (3-sentence summary + link)
- Computer vision filter panel: RGB channel sliders, Canny edge detection, Grayscale, Adaptive Thresholding
- Geographic distribution visualisation (PyDeck)
- Quiz mode: multiple-choice questions generated from model prediction classes

---

## 6. Limitations & Future Work

The current training regime fine-tunes only the classification head. Unfreezing the final 2–3 backbone blocks with a reduced learning rate (≈1e-4) would allow the feature extractor to adapt to animal-specific textures, likely recovering 2–4 additional accuracy points. The stacked ensemble should be retrained with base models frozen in `.eval()` mode and, ideally, using out-of-fold predictions to avoid implicit label leakage from shared validation statistics.

Chronic weak classes (moth, donkey, ox, possum) may benefit from targeted data augmentation or the use of a hierarchical loss that penalises within-cluster confusions more heavily than cross-cluster ones.

---

## 7. Repository Structure

```
├── train.py                  # Full training pipeline (all three models)
├── metrics.py                # Evaluation: curves, confusion matrices, reports
├── app.py                    # Streamlit inference application
├── checkpoints/              # Saved model weights (.pth)
└── results/
    ├── MobileNetV2/
    ├── EfficientNetB0/
    └── StackedEnsemble/
```

---

## 8. Reproduction

```bash
pip install torch torchvision streamlit opencv-python-headless \
            pandas numpy scikit-learn matplotlib pydeck wikipedia kagglehub

# Train all models and auto-generate metrics
python train.py

# Run inference app locally
streamlit run app.py

# Generate metrics from existing checkpoints (standalone)
python metrics.py \
  --mobilenet    checkpoints/mobilenet_v2_full.pth \
  --efficientnet checkpoints/efficientnet_full.pth \
  --ensemble     checkpoints/stacked_ensemble_full.pth \
  --dataset      /path/to/animals/animals
```

---

## 9. Tech Stack

PyTorch · Torchvision · Streamlit · Hugging Face Spaces · OpenCV · scikit-learn · NumPy · Pandas · Wikipedia API · PyDeck · Matplotlib · KaggleHub
