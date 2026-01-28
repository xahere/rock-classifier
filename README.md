# Lithos AI - Rock Classification System

Lithos AI is a deep learning-based application designed to classify geological rock samples into three primary categories: **Igneous**, **Metamorphic**, and **Sedimentary**. 

The project features a Flask-based backend powered by Tensorflow/Keras (MobileNetV2) and a modern, responsive frontend using vanilla JavaScript and CSS.

## Features

- **Accurate Classification**: Uses a fine-tuned MobileNetV2 model trained on ~3,500 rock images.
- **Robust Pipeline**: Automated dataset aggregation from multiple sources (GitHub).
- **User-Friendly UI**: Drag-and-drop interface with glassmorphism design.
- **Offline Capabilities**: Runs entirely locally on CPU/GPU.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/xahere/rock-classifier.git
   cd lithos-ai
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Dataset (First Run Only)
The project includes a utility script to clone and organize the necessary training data.
```bash
python dataset_loader.py
```

### 2. Train the Model (Optional)
If you want to re-train the model from scratch:
```bash
python train_model.py
```
*Note: This process uses Transfer Learning and Fine-Tuning. Training time depends on your hardware.*

### 3. Run the Application
Start the Flask server:
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

## Architecture
- **Model**: MobileNetV2 (Transfer Learning, ImageNet weights)
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (No frameworks)
- **Data Augmentation**: Rotation, Zoom, Shear, Horizontal/Vertical Flip.

## License
MIT License
