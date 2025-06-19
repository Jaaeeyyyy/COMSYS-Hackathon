# 🧠 Task A: Gender Classification – FACECOM Hackathon

## 📌 Overview

This repository contains the solution to **Task A: Gender Classification** from the **Comsys FACECOM Hackathon**. The objective was to develop a deep learning-based binary classification model that distinguishes between **male** and **female** faces under visually degraded conditions.

> **Participant:** Jay Maheshwari

---

## 📊 Final Results

| Metric       | Score    |
|--------------|----------|
| Accuracy     | 92.89%   |
| Precision    | 93.14%   |
| Recall       | 92.89%   |
| F1-Score     | 92.99%   |

### 📈 Class-wise Accuracy:
- **Female:** 84.81% (67/79 samples)
- **Male:** 94.75% (325/343 samples)

---

## 🏗️ Model Details

- **Backbone:** `EfficientNet-B0` (pretrained on ImageNet)
- **Classifier Head:** Modified for binary output
- **Input Resolution:** `224×224` (RGB)
- **Framework:** PyTorch

---

## 📁 Dataset Structure

```
FACECOM/Task_A/
├── train/
│   ├── female/
│   └── male/
└── val/
    ├── female/
    └── male/
```

- **Training Samples:** 1,926
- **Validation Samples:** 422
- **Class Distribution:** Female – 303, Male – 1,623 (Imbalanced)

---

## 🧰 Key Components

### 🧼 Data Preprocessing & Augmentation
- **Training:**
  - Random horizontal flip (p=0.5)
  - Rotation ±10°
  - Color jitter (brightness, contrast, saturation, hue)
  - Random affine transformations
  - ImageNet normalization
- **Validation:** Resize and normalize (no augmentation)

### ⚖️ Imbalance Handling
- **Weighted Random Sampler** to mitigate skewed class distribution
- Class weights computed using inverse frequency

### 🧪 Training Setup
- **Loss:** `CrossEntropyLoss`
- **Optimizer:** `AdamW` (weight decay = 0.01)
- **Learning Rate:** `0.001` (with `ReduceLROnPlateau`)
- **Early Stopping:** Patience = 10 epochs
- **Batch Size:** 32

---

## 🚀 Setup & Installation

### 🔧 Requirements

```bash
pip install torch torchvision scikit-learn numpy
```

### 📄 Libraries Used
- `torch`, `torchvision`
- `scikit-learn`
- `numpy`

---

## 💻 How to Use

### 1. Training

```python
from google.colab import drive
drive.mount('/content/drive')  # For Colab

# Update with actual dataset path
train_dir = '/path/to/FACECOM/Task_A/train'
val_dir = '/path/to/FACECOM/Task_A/val'

# Run training
python train_gender_classifier.py
```

### 2. Evaluation
After training, the script:
- Computes overall metrics
- Generates a detailed classification report
- Displays per-class accuracy

### 3. Inference

```python
import torch
from torchvision import models
import torch.nn as nn

checkpoint = torch.load('final_gender_classification_model.pth')
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Add your inference logic below
```

---

## 📉 Training Summary

- **Epochs Trained:** 13 (Early Stopping)
- **Best Epoch:** 3
- **Training Accuracy:** 96.31%
- **Validation Accuracy:** 92.89%
- **Training Loss:** 0.1095
- **Validation Loss:** 0.2351

---

## 📈 Performance Analysis

### ✅ Strengths
- High overall accuracy and strong male classification performance
- Effective class imbalance mitigation via weighted sampling
- Early stopping to avoid overfitting

### ⚠️ Areas for Improvement
- Female classification performance (~85%) can be improved
- Limited number of female samples
- Additional augmentation or data synthesis could help

---

## 🧪 Technical Breakdown

### 📸 Augmentation Pipeline
- Geometric transforms (rotation, affine)
- Color distortions (jitter)
- Horizontal flips

### ⚖️ Imbalance Strategy
- 5:1 male-to-female ratio handled using a `WeightedRandomSampler`
- Inverse frequency weighting ensured balanced mini-batches

### ⏹️ Early Stopping
- Monitors validation accuracy
- Saves best checkpoint
- Triggers after 10 stagnant epochs

---

## 🏁 Hackathon Outcome

Achieved **92.89%** validation accuracy — outperforming the reported baseline (>90.28%) in the challenge.

---

## 🙌 Acknowledgments

- **Event:** Comsys FACECOM Hackathon
- **Dataset:** FACECOM Dataset
- **Frameworks:** PyTorch, torchvision
- **Backbone Model:** EfficientNet-B0 (ImageNet weights)

---

## 📬 Contact

**Jay Maheshwari**  
- GitHub: [https://github.com/Jaaeeyyyy](https://github.com/Jaaeeyyyy)  
- LinkedIn: [https://linkedin.com/in/jay-maheshwari-6b8109250](https://linkedin.com/in/jay-maheshwari-6b8109250)  
- Email: jaymaheshwari2208@gmail.com

---

**Note:** This solution was developed as part of the Comsys Hackathon. It is clean, reproducible, and optimized for performance under real-world constraints.
