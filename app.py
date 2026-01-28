import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = "rock_classifier.h5"
CLASS_INDICES_PATH = "class_indices.txt"

# Globals for inference
model = None
class_names = {}

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
    Returns JSON with 'class' and 'confidence'.
    """
    if not model:
        return jsonify({'error': 'Model not initialized'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
            
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'class': result_class,
                'confidence': f"{confidence:.2%}"
            })
            
        except Exception as e:
            # Ensure cleanup even on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_inference_model()
    app.run(debug=True, port=5000, host='0.0.0.0')
