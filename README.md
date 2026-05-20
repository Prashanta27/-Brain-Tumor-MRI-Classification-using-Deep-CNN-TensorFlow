# 🧠 Brain Tumor MRI Classification using Deep CNN + TensorFlow

## 🚀 Project Overview

Medical image classification is one of the most impactful applications of Deep Learning in healthcare.  
In this project, I built a Convolutional Neural Network (CNN) model that classifies brain MRI scans into 4 categories:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

Dataset used: Kaggle Brain Tumor MRI Dataset

## 📊 Results

- ✅ Training Accuracy: 93%
- ✅ Validation Accuracy: 92%
- ✅ Training Loss: ~7%
- ✅ Validation Loss: ~8%

The model shows strong generalization with minimal overfitting.

---

## 📌 Problem Statement

Brain tumor detection is a critical medical task. Manual MRI analysis is time-consuming and requires expert radiologists.

This project aims to automate MRI classification using Deep Learning for faster and more accurate diagnosis support.

---

## 🧬 Dataset Classes

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- PIL

---

## 🧠 Model Architecture

The CNN model includes:

### 🔹 Data Augmentation
- Random Flip
- Random Rotation
- Random Zoom

### 🔹 Convolution Blocks
Each block:
- Conv2D
- Batch Normalization
- ReLU Activation
- MaxPooling
- Dropout

### 🔹 Feature Extraction
- Global Average Pooling

### 🔹 Classification Head
- Dense (256)
- Dropout (0.5)
- Softmax Output Layer

---

## ⚡ Key Design Choices

### ✔ Why Batch Normalization?
Stabilizes training and improves convergence.

### ✔ Why Dropout?
Prevents overfitting and improves generalization.

### ✔ Why Global Average Pooling?
Reduces parameters compared to Flatten and avoids overfitting.

### ✔ Why AdamW Optimizer?
Combines adaptive learning with weight decay for better generalization.

---

## 📉 Loss Function

Sparse Categorical Crossentropy used for multi-class classification.

---

## 🛑 Training Strategy

- EarlyStopping (patience = 5)
- ReduceLROnPlateau
- Best weight restoration enabled

---

## 📊 Evaluation

### Classification Report
Precision, Recall, and F1-score used for performance measurement.

### Confusion Matrix
Shows model performance across all 4 classes.

---

## 💾 Model Saving

```python
model.save("Brain_tumor_cnn.keras")
