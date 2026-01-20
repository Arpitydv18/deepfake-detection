# Deepfake Detection Using Deep Learning

## 📌 Project Overview

Deepfake Detection is a deep learning–based computer vision project that identifies whether a given face image is **Real** or **Fake (Deepfake)**. The model learns subtle visual patterns from large-scale image data to detect manipulated media and help combat misinformation.

---

## 🚀 Features

* Binary classification: **Real vs Fake**
* CNN-based deep learning model
* Data augmentation for better generalization
* Model evaluation using Accuracy, AUC, ROC Curve, and Confusion Matrix
* Interactive **Gradio web interface** for real-time predictions
* GPU-accelerated training (NVIDIA RTX 3050)

---

## 🧠 Tech Stack

* **Language:** Python
* **Frameworks:** TensorFlow, Keras
* **Libraries:** NumPy, OpenCV, Matplotlib, Scikit-learn
* **Frontend (Demo):** Gradio
* **Hardware:** NVIDIA GPU (CUDA + cuDNN)

---

## 📂 Dataset Structure

```
deepfake_Dataset/
│── Train/
│   ├── Real/
│   └── Fake/
│── Validation/
│   ├── Real/
│   └── Fake/
│── Test/
│   ├── Real/
│   └── Fake/
```

* **Training samples:** 140,002
* **Validation samples:** 39,428
* **Test samples:** 10,905

---

## 🏗️ Model Architecture

* Convolutional Neural Networks (CNN)
* Batch Normalization
* Max Pooling
* Dropout for regularization
* Softmax output layer (2 classes)

The model was trained using **categorical cross-entropy loss** and the **Adam optimizer** with learning rate scheduling.

---

## ⚙️ Training Configuration

* Image Size: `256 × 256`
* Batch Size: `32`
* Epochs: `50`
* Optimizer: `Adam`
* Callbacks:

  * ModelCheckpoint
  * EarlyStopping
  * ReduceLROnPlateau
  * TensorBoard

---

## 📊 Model Performance

**Test Set Results:**

* **Accuracy:** 82.7%
* **AUC:** 0.913

**Classification Summary:**

* Real: High recall (92%)
* Fake: High precision (90%)

ROC curve and confusion matrix were used to analyze model performance and class balance.

---

## 🧪 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* ROC Curve
* Confusion Matrix

---

## 🖼️ Sample Predictions

The model successfully predicts unseen images as **Real** or **Fake**, demonstrating strong generalization capability.

---

## 🌐 Gradio Web Interface

A user-friendly Gradio interface allows users to:

* Upload or capture a face image
* Instantly detect whether the image is Real or Fake

```python
interface.launch(share=True)
```

---

## 💾 Model Saving

The trained model is saved in `.h5` format:

```
deepfake_detection_model.h5
```

---

## 🎯 Key Learnings

* End-to-end ML pipeline development
* Handling large-scale image datasets
* CNN optimization and regularization
* GPU-based training and memory management
* Model evaluation using real-world metrics
* Deploying ML models using Gradio

---

## 📌 Future Improvements

* Video-based deepfake detection
* Transformer-based architectures
* Improved generalization across datasets
* Deployment using Docker / Cloud platforms

---

## 👤 Author

**Arpit Yadav**
Machine Learning | Deep Learning | Computer Vision

---

⭐ If you like this project, feel free to star the repository!
