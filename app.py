"""
Lithos AI - Rock Classification Server.

This Flask application serves the trained MobileNetV2 model for classifying rock images
into Igneous, Metamorphic, and Sedimentary categories. It handles image uploads,
preprocessing, and inference. It also supports an Adaptive Feedback Loop for self-learning.
"""

import os
import shutil
import threading
import numpy as np
import time
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from werkzeug.utils import secure_filename
import train_model  # Import our training module

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATASET_FOLDER'] = 'dataset_organized'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = "rock_classifier.h5"
CLASS_INDICES_PATH = "class_indices.txt"

# Globals for inference and training state
model = None
class_names = {}
is_training = False

def load_inference_model():
    """Loads the pre-trained model and class mappings into memory."""
    global model, class_names
    
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading model from {MODEL_PATH}...")
            model = load_model(MODEL_PATH)
            print("Model loaded successfully.")
            
            if os.path.exists(CLASS_INDICES_PATH):
                with open(CLASS_INDICES_PATH, "r") as f:
                    # Clear existing to avoid dupes on reload
                    class_names = {}
                    for line in f:
                        parts = line.strip().split(":")
                        if len(parts) == 2:
                            class_names[int(parts[0])] = parts[1]
            else:
                 # Default fallback
                 class_names = {0: 'Igneous', 1: 'Metamorphic', 2: 'Sedimentary'}
                 
        except Exception as e:
            print(f"Critical error loading model: {e}")
    else:
        print(f"Warning: Model file '{MODEL_PATH}' not found. Prediction endpoint will fail.")

@app.route('/')
def index():
    """Serves the main frontend page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API access point for image prediction.
    Expects a multipart/form-data upload with key 'file'.
    Returns JSON with 'class', 'confidence', 'filename'.
    """
    global model
    if not model:
        return jsonify({'error': 'Model not initialized'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        # Add timestamp to filename to ensure uniqueness for feedback
        unique_filename = f"{int(time.time())}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            # Preprocess for MobileNetV2 (1./255 scaling)
            img = image.load_img(filepath, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0
            
            preds = model.predict(x)
            pred_class_idx = np.argmax(preds, axis=1)[0]
            confidence = float(np.max(preds))
            
            result_class = class_names.get(pred_class_idx, "Unknown")
            
            # We DO NOT delete the file here anymore, we wait to see if user gives feedback
            # If no feedback, we can clean up later (e.g. cron job), or client sends delete req
            # For now, we return the server-side filename so client can ref it in feedback
            
            return jsonify({
                'class': result_class,
                'confidence': f"{confidence:.2%}",
                'filename': unique_filename
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """
    Endpoint for Adaptive Learning.
    Receives: { 'filename': '...', 'correct_label': 'Igneous' }
    Action: Moves file to dataset, augments it 5x, AND TRIGGERS RETRAINING.
    """
    data = request.json
    filename = data.get('filename')
    correct_label = data.get('correct_label')
    
    if not filename or not correct_label:
        return jsonify({'error': 'Missing data'}), 400
        
    src_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(src_path):
        return jsonify({'error': 'File not found. May have expired.'}), 404
        
    # Validate label
    valid_labels = ['Igneous', 'Metamorphic', 'Sedimentary']
    if correct_label not in valid_labels:
        return jsonify({'error': 'Invalid label'}), 400

    # Dest folder
    dest_dir = os.path.join(app.config['DATASET_FOLDER'], correct_label)
    os.makedirs(dest_dir, exist_ok=True)
    
    # 1. Move the original file
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(src_path): 
        shutil.move(src_path, dest_path)
    
    # 2. Augment (Generate 5 variations)
    try:
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        img = load_img(dest_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        
        i = 0
        for batch in datagen.flow(x, batch_size=1, 
                                  save_to_dir=dest_dir, 
                                  save_prefix=f"aug_{int(time.time())}", 
                                  save_format='jpg'):
            i += 1
            if i >= 5: 
                break
                
        # 3. SILENT TRIGGER: Auto-start retraining background thread
        def run_retraining():
            global is_training, model
            if is_training: return # Skip if already running
            
            is_training = True
            try:
                print("Starting SILENT background retraining...")
                train_model.retrain_lite()
                print("Silent retraining complete. Reloading model...")
                load_inference_model()
            except Exception as e:
                print(f"Retraining failed: {e}")
            finally:
                is_training = False
                
        thread = threading.Thread(target=run_retraining)
        thread.start()

        return jsonify({'message': 'Feedback accepted. Model is learning from this pattern in the background.'})
        
    except Exception as e:
        print(f"Augmentation error: {e}")
        return jsonify({'error': 'File moved, but augmentation/retraining failed.'}), 500

# Endpoint kept for manual admin admin usage if needed, but UI will rely on auto-trigger
@app.route('/retrain', methods=['POST'])
def retrain():
    # ... (Logic identical to internal trigger)
    return jsonify({'status': 'ignored', 'message': 'Use feedback loop for auto-training.'})

if __name__ == '__main__':
    load_inference_model()
    app.run(debug=True, use_reloader=False, port=5000, host='0.0.0.0')
