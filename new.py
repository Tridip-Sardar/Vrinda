# cure_api.py - Fixed version with improved error handling
import tensorflow as tf
import numpy as np
import requests
import json
from flask import Flask, request, jsonify
import os
from PIL import Image
import time
import logging
from werkzeug.utils import secure_filename
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Global variable to store model
plant_model = None

# Class names (fixed formatting)
class_name = [
    'Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
    'Blueberry__healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy', 
    'Corn_(maize)Cercospora_leaf_spot_Gray_leaf_spot', 'Corn(maize)Common_rust', 
    'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape__Black_rot', 
    'Grape_Esca(Black_Measles)', 'GrapeLeaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy', 
    'Orange_Haunglongbing(Citrus_greening)', 'PeachBacterial_spot', 'Peach_healthy', 
    'Pepper,bell_Bacterial_spot', 'Pepper,bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight', 
    'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 
    'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 
    'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two-spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato_Tomato_mosaic_virus', 'Tomato_healthy'
]

# Ollama Configuration
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-r1:1.5b"
FAST_MODELS = ["llama3.2:1b", "phi3.5:3.8b", "gemma2:2b"]

def load_model():
    """Load the trained model with error handling"""
    global plant_model
    model_path = 'keras_model/trained_model.keras'
    
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        plant_model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        plant_model = None
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_ollama_response(prompt, max_retries=2, timeout=45):
    """Get response from Ollama API with optimized settings"""
    simplified_prompt = f"""Provide concise treatment and prevention advice for the plant disease: {prompt.replace('What is the cure and precaution for ', '')}

Include:
- Primary treatment methods
- Prevention strategies
- Key symptoms to watch for

Keep response under 250 words."""

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                headers={"Content-Type": "application/json"},
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": simplified_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 200,
                        "top_k": 40,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                        "num_ctx": 1024
                    }
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "").strip()
                
                if response_text and len(response_text) > 20:
                    return response_text
                else:
                    continue
            else:
                logger.warning(f"Ollama API returned status code: {response.status_code}")
                continue
                
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection failed (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2)
        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            logger.error(f"Error with Ollama: {e}")
    
    return None

def get_fallback_response(disease_name):
    """Provide fallback treatment information when Ollama is unavailable"""
    disease_lower = disease_name.lower()
    
    if 'healthy' in disease_lower:
        return {
            "treatment": "Plant appears healthy. Continue current care routine.",
            "prevention": "Maintain proper watering, nutrition, and monitor regularly for changes.",
            "symptoms": "No disease symptoms detected."
        }
    
    elif any(term in disease_lower for term in ['bacterial', 'spot', 'blight']):
        return {
            "treatment": "Remove affected leaves immediately. Apply copper-based fungicide. Improve air circulation.",
            "prevention": "Water at soil level, space plants properly, remove plant debris, practice crop rotation.",
            "symptoms": "Dark spots on leaves, yellowing, wilting of affected areas."
        }
    
    elif any(term in disease_lower for term in ['rust', 'mildew', 'mold']):
        return {
            "treatment": "Apply fungicidal spray. Remove infected plant parts. Increase air circulation.",
            "prevention": "Avoid watering leaves, ensure proper spacing, use disease-resistant varieties.",
            "symptoms": "Powdery coating, orange/rust colored spots, fuzzy growth on leaves."
        }
    
    elif 'virus' in disease_lower:
        return {
            "treatment": "Remove infected plants immediately. Control insect vectors. No chemical cure available.",
            "prevention": "Use virus-free seeds, control aphids and vectors, remove weeds that harbor viruses.",
            "symptoms": "Yellowing, stunted growth, distorted leaves, mosaic patterns."
        }
    
    else:
        return {
            "treatment": "Remove affected plant parts. Apply appropriate pesticide/fungicide based on disease type.",
            "prevention": "Maintain plant health, ensure proper drainage, use disease-resistant varieties.",
            "symptoms": "Consult agricultural extension service for specific symptom identification."
        }

def check_ollama_status():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def model_prediction(image_path):
    """Make prediction using the trained model"""
    if plant_model is None:
        raise ValueError("Model not loaded")
    
    try:
        # Load and preprocess image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image = image.resize((128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr]) / 255.0  # Normalize
        
        prediction = plant_model.predict(input_arr)
        result_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        return result_index, confidence
        
    except Exception as e:
        logger.error(f"Error in model prediction: {str(e)}")
        raise

@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint"""
    if plant_model is None:
        return jsonify({
            "error": "Model not loaded",
            "details": "The ML model failed to load. Please check the model file path.",
            "status": "error"
        }), 500
    
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    
    if image_file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    if not allowed_file(image_file.filename):
        return jsonify({
            "error": "Invalid file type",
            "allowed_types": list(ALLOWED_EXTENSIONS)
        }), 400

    try:
        # Generate unique filename
        filename = secure_filename(image_file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Save uploaded image
        image_file.save(file_path)
        logger.info(f"Image saved to {file_path}")

        # Predict disease
        result_index, confidence = model_prediction(file_path)
        predicted_disease = class_name[result_index].replace("_", " ")
        
        logger.info(f"Predicted disease: {predicted_disease} (confidence: {confidence:.3f})")

        # Check if Ollama is available
        ollama_available = check_ollama_status()
        
        if ollama_available:
            # Try to get response from Ollama
            prompt = f"What is the cure and precaution for {predicted_disease}?"
            cure_text = get_ollama_response(prompt)
            
            if cure_text:
                response_data = {
                    "predicted_disease": predicted_disease,
                    "confidence_score": round(confidence, 3),
                    "cure_and_precaution": cure_text,
                    "confidence": "AI-generated",
                    "source": "Ollama AI",
                    "status": "success"
                }
            else:
                # Ollama failed, use fallback
                fallback = get_fallback_response(predicted_disease)
                response_data = {
                    "predicted_disease": predicted_disease,
                    "confidence_score": round(confidence, 3),
                    "cure_and_precaution": f"**Treatment:** {fallback['treatment']}\n\n**Prevention:** {fallback['prevention']}\n\n**Symptoms:** {fallback['symptoms']}",
                    "confidence": "Expert knowledge base",
                    "source": "Fallback system",
                    "status": "fallback_used",
                    "note": "AI service unavailable, using expert knowledge base"
                }
        else:
            # Ollama not available, use fallback immediately
            fallback = get_fallback_response(predicted_disease)
            response_data = {
                "predicted_disease": predicted_disease,
                "confidence_score": round(confidence, 3),
                "cure_and_precaution": f"**Treatment:** {fallback['treatment']}\n\n**Prevention:** {fallback['prevention']}\n\n**Symptoms:** {fallback['symptoms']}",
                "confidence": "Expert knowledge base",
                "source": "Fallback system", 
                "status": "ollama_unavailable",
                "note": "Ollama service not available, using expert knowledge base"
            }

        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")

        return jsonify(response_data)

    except Exception as e:
        # Clean up temporary file in case of error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
            
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({
            "error": "Prediction failed",
            "details": str(e),
            "status": "error"
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    ollama_status = check_ollama_status()
    model_status = plant_model is not None
    
    return jsonify({
        "status": "running",
        "model_loaded": model_status,
        "ollama_available": ollama_status,
        "model": OLLAMA_MODEL,
        "host": OLLAMA_HOST,
        "fallback_enabled": True
    })

@app.route("/models", methods=["GET"])
def list_models():
    """List available Ollama models"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return jsonify({
                "available_models": [m.get("name", "Unknown") for m in models],
                "current_model": OLLAMA_MODEL,
                "recommended_fast_models": FAST_MODELS
            })
        else:
            return jsonify({"error": "Could not fetch models from Ollama"}), 500
    except Exception as e:
        return jsonify({"error": f"Ollama service not available: {str(e)}"}), 503

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Please upload an image smaller than 16MB."}), 413

@app.errorhandler(415)
def unsupported_media_type(e):
    return jsonify({"error": "Unsupported file type. Please upload a valid image file."}), 415

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "details": str(e)}), 500

# Cleanup function for temporary files
def cleanup_old_files():
    """Clean up old temporary files"""
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            # Remove files older than 1 hour
            if os.path.isfile(file_path) and time.time() - os.path.getctime(file_path) > 3600:
                os.remove(file_path)
                logger.info(f"Cleaned up old file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")

if __name__ == "__main__":
    print("🌱 Plant Disease Cure API Starting...")
    print(f"🤖 Using Ollama model: {OLLAMA_MODEL}")
    print(f"🌐 Ollama host: {OLLAMA_HOST}")
    
    # Load the model
    if load_model():
        print("✅ ML model loaded successfully")
    else:
        print("❌ Failed to load ML model - API will not work properly")
    
    # Check Ollama status on startup
    if check_ollama_status():
        print("✅ Ollama service is available")
    else:
        print("⚠️ Ollama service not available - will use fallback responses")
    
    # Clean up any existing temporary files
    cleanup_old_files()
    
    print("🚀 Starting Flask server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)