# Lithos AI - Rock Classification System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-Web%20Server-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Lithos AI** is a lightweight, offline-capable deep learning application designed to classify geological rock samples into three primary petrological categories: **Igneous**, **Metamorphic**, and **Sedimentary**.

Powered by a fine-tuned **MobileNetV2** architecture, it brings the power of Transfer Learning to standard laptop CPUs, making it an ideal tool for geology students and field researchers working in remote environments.

![Lithos AI Homepage](https://github.com/xahere/rock-classifier/blob/main/lithos_ai_homepage_1768416817355.png?raw=true)
*(Note: Replace the above link with your actual screenshot path if hosted differently)*

---

## 🚀 Key Features

*   **Offline First**: Designed to run entirely on a local machine (localhost) without internet dependency.
*   **Adaptive Feedback Loop**: **Self-learning AI** that accepts user corrections, augments data, and **silently retrains** in the background to improve over time.
*   **High Efficiency**: Optimized **MobileNetV2** backend allows for sub-second inference on standard CPUs.
*   **Glassmorphism UI**: A modern, responsive interface built with vanilla HTML/CSS/JS for a premium user experience.
*   **Retraining Pipeline**: Automated background thread handles data augmentation and model fine-tuning without disrupting the user.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/xahere/rock-classifier.git
cd rock-classifier
```

### 2. Set Up Environment
It is recommended to use a virtual environment to manage dependencies.
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Ensure you have Python 3.9+ installed)*

---

## 💻 Usage

### 📊 Dataset Setup
The project comes with a helper script to organize your data.
```bash
python dataset_loader.py
```
*This will structure your raw images into `dataset_organized/` format for training.*

### 🧠 Model Training (Administrator Only)
To retrain the model with your own custom dataset:
1.  Place your new images into the respective folders in `dataset_organized/`.
2.  Run the training script:
```bash
python train_model.py
```
*This uses Transfer Learning to fine-tune the model, achieving ~65% accuracy on standard benchmarks.*

### 🚀 Run the Application
Start the local server for inference:
```bash
python app.py
```
Access the interface at: **http://127.0.0.1:5000**

---

## 📁 Project Structure

```text
rock-classifier/
├── app.py                  # Main Flask Application Entry Point
├── train_model.py          # Training Pipeline (MobileNetV2)
├── dataset_loader.py       # ETL Utility for Data Management
├── rock_classifier.h5      # Trained Model Weights (Generated)
├── class_indices.txt       # Label Mapping
├── requirements.txt        # Python Dependencies
├── static/
│   ├── style.css           # Glassmorphism Styling
│   └── script.js           # Frontend Logic (Drag & Drop)
└── templates/
    └── index.html          # Main User Interface
```

---

## 🔬 Results & Performance

*   **Architecture**: MobileNetV2 (Pre-trained on ImageNet).
*   **Validation Accuracy**: ~64-65% (on 3-class split).
*   **Inference Time**: <100ms per image (CPU).
*   **Limitations**: The model is currently optimized for broad families (Igneous/Metamorphic/Sedimentary) and may require distinct textural features (banding, grains) for high confidence.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
