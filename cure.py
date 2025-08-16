import tensorflow as tf
import numpy as np
import streamlit as st
import requests
import json
import os
import time
from pathlib import Path

# CSS Styling with Custom UI Elements (keeping your existing styles)
css_code = """
:root {
  /* Colors */
  --body-color: #e4f7ee;  /* Set the background color here */
  --sidebar-color: #fff;
  --primary-color: #008080;
  --primary-color-light: #f6f5ff;
  --toggle-color: #ddd;
  --text-color: #707070;
}

/* Hide Streamlit's deploy button and menu */
header, footer, .css-1lsmgbg.e1fqkh3o3, .css-9s5bis.edgvbvh10 {
  display: none !important;
}

/* Body Background */
body {
  background-color: var(--body-color);  /* Apply background color here */
  color: var(--text-color);
  background-image: url("/home/tridip/Desktop/Plant_disease_backend_copy/bg/pine-leaf.png");
}
.stButton button {
  background-color: #008080 !important;  /* Force the color for Streamlit's button */
  border: 2px solid #008080 !important;  /* Force the border color */
}
/* Predict button hover effect */
.stButton button:hover {
  background-color: teal !important;  /* Change background color on hover */
  color: white !important;  /* Ensure font color remains white on hover */
  box-shadow: 0px 4px 8px rgba(1, 179, 179, 0.48);
}

/* Upload Container */
.uploaded-div {
    font-size: 1.5rem;
  padding: 2rem 0.5rem;
  text-align: center;
  width: 100%;
  max-width: 20rem;
  margin: 2rem auto;
}

/* Buttons */
button, .browse-btn, .predict-btn {
  padding: 10px 20px;
  font-size: 16px;
  font-weight: 600;
  color: #fff;
  background-color: var(--primary-color);
  border: 2px solid teal;
  border-radius: 12px;
  transition: 0.3s;
  text-align: center;
  cursor: pointer;
}

/* Update color for predict button specifically */
.predict-btn {
  background-color: #008080;  /* Change to desired color */
  border: 2px solid #008080;  /* Match the border to the button color */
}

.browse-btn:hover, .predict-btn:hover {
  background-color: teal;
  box-shadow: 0px 4px 8px rgba(1, 179, 179, 0.48);
}

/* Prediction Result */
.result-text {
  font-size: 22px;
  color: teal;
  font-weight: 600;
  margin: 1rem auto;
  text-align: center;
}
/* Buttons */
button, .browse-btn, .predict-btn {
  padding: 10px 20px;
  font-size: 16px;
  font-weight: 600;
  color: #fff;  /* Set the default font color to white */
  background-color: var(--primary-color);
  border: 2px solid teal;
  border-radius: 12px;
  transition: 0.3s;
  text-align: center;
  cursor: pointer;
}
/* Image Preview */
.preview {
  width: 100%;
  max-width: 400px;
  margin: 1rem auto;
  border: 1px dashed black;
  border-radius: 12px;
}

.preview img {
  width: 100%;
  height: auto;
  border-radius: 12px;
}

/* Loading spinner */
.loading-spinner {
  text-align: center;
  padding: 20px;
  color: teal;
}

/* Error message styling */
.error-message {
  color: #d32f2f;
  background-color: #ffebee;
  padding: 10px;
  border-radius: 8px;
  border-left: 4px solid #d32f2f;
  margin: 10px 0;
}
"""

# Apply the CSS styles to Streamlit
st.markdown(f"<style>{css_code}</style>", unsafe_allow_html=True)

# Ollama Configuration
OLLAMA_HOST = "http://localhost:11434"  # Using the working port from your setup
OLLAMA_MODEL = "deepseek-r1:1.5b"

def get_ollama_response(prompt, max_retries=2):
    """
    Get response from Ollama API with error handling and retries
    """
    # Shorter, more focused prompt to reduce processing time
    simplified_prompt = f"""Provide treatment and prevention for {prompt.split(':')[1].strip() if ':' in prompt else prompt}. 
Keep response under 300 words. Include:
- Treatment methods
- Prevention tips
- Key symptoms"""
    
    for attempt in range(max_retries):
        try:
            st.info(f"🤖 Generating response... (attempt {attempt + 1})")
            
            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                headers={"Content-Type": "application/json"},
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": simplified_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for faster, more focused responses
                        "num_predict": 200,   # Limit response length
                        "top_k": 40,         # Reduce choices for faster generation
                        "top_p": 0.9,        # Focus on most likely tokens
                        "repeat_penalty": 1.1,
                        "num_ctx": 1024      # Smaller context window
                    }
                },
                timeout=60  # Increased timeout to 60 seconds
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "No response generated")
                if response_text and len(response_text.strip()) > 10:
                    return response_text
                else:
                    st.warning("⚠️ Received empty response, retrying...")
                    continue
            else:
                st.error(f"Ollama API returned status code: {response.status_code}")
                if attempt < max_retries - 1:
                    continue
                return None
                
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                st.warning(f"🔌 Connection failed, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(3)
            else:
                st.error("❌ Cannot connect to Ollama. Please check if Ollama is running.")
                return None
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                st.warning(f"⏱️ Request timed out, trying shorter prompt... (attempt {attempt + 1}/{max_retries})")
                # Try with an even shorter prompt
                simplified_prompt = f"Treatment for {prompt.split(':')[1].strip() if ':' in prompt else prompt}?"
                time.sleep(2)
            else:
                st.error("⏱️ Request consistently timing out. Try using a smaller/faster model.")
                return None
        except Exception as e:
            st.error(f"Error communicating with Ollama: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None
    
    return None

def check_ollama_connection():
    """
    Check if Ollama is running and the model is available
    """
    try:
        # Check if Ollama is running
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            # Check if our model is available
            model_available = any(OLLAMA_MODEL in name for name in model_names)
            
            if model_available:
                return True, "✅ Ollama is running and model is available"
            else:
                return False, f"❌ Model '{OLLAMA_MODEL}' not found. Available models: {model_names}"
        else:
            return False, f"❌ Ollama responded with status: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return False, "❌ Cannot connect to Ollama. Please start Ollama service."
    except Exception as e:
        return False, f"❌ Error checking Ollama: {str(e)}"

# Function to load the model and make predictions
def model_prediction(test_image):
    model = tf.keras.models.load_model('keras_model/trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Class labels (keeping your existing labels)
class_name = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy', 
    'Corn_(maize)Cercospora_leaf_spot_Gray_leaf_spot', 'Corn(maize)Common_rust', 
    'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape__Black_rot', 
    'Grape__Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange__Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy', 
    'Pepper,bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Potato__Late_blight', 
    'Potato__healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash__Powdery_mildew', 
    'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato__Early_blight', 
    'Tomato__Late_blight', 'Tomato_Leaf_Mold', 'Tomato__Septoria_leaf_spot', 
    'Tomato__Spider_mites_Two-spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato__Tomato_mosaic_virus', 'Tomato__healthy'
]

# Main App UI
st.markdown('<h1 style="text-align:center;">🌱 Plant Disease Recognition</h1>', unsafe_allow_html=True)

# Check Ollama connection at startup
connection_status, status_message = check_ollama_connection()
if connection_status:
    st.success(status_message)
else:
    st.error(status_message)
    st.info("💡 To start Ollama, run: `ollama serve` in your terminal")
    
    # Show troubleshooting info
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **If you see connection errors:**
        1. Make sure Ollama is installed and running: `ollama serve`
        2. Check if the model is available: `ollama list`
        3. If the model is missing, install it: `ollama pull deepseek-r1:1.5b`
        4. Verify Ollama is running on port 11434: `curl http://localhost:11434/api/tags`
        """)

st.markdown('<div class="uploaded-div">Upload an Image of the Plant</div>', unsafe_allow_html=True)
test_image = st.file_uploader("Choose an Image")

# Add loading bar while the image is being uploaded
if test_image is not None:
    # Simulate a file upload with a loading bar
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.02)  # Simulating a loading time
        progress.progress(i + 1)
    
    st.image(test_image, width=400)

if st.button("Predict"):
    if test_image is not None:
        if not connection_status:
            st.error("❌ Cannot make prediction: Ollama is not available")
        else:
            # Make prediction
            with st.spinner("🔍 Analyzing the plant image..."):
                result_index = model_prediction(test_image)
                predicted_disease = class_name[result_index]
            
            st.success(f"Model predicts this image is of a plant with: **{predicted_disease}**")
            
            # Get cure and precautions from Ollama with fallback options
            st.markdown(f'<div class="result-text">🌿 Getting Expert Advice for {predicted_disease}...</div>', unsafe_allow_html=True)
            
            # Try to get AI response
            response_text = get_ollama_response(f"Disease: {predicted_disease}")
            
            if response_text:
                st.success("✅ AI Response Generated!")
                st.markdown("### 🩺 Treatment & Prevention")
                st.markdown(response_text)
            else:
                # Provide comprehensive fallback information
                st.warning("⚠️ AI service unavailable. Showing general guidance:")
                
                # Basic treatment advice based on disease type
                disease_lower = predicted_disease.lower()
                
                if 'healthy' in disease_lower:
                    fallback_text = """
**✅ Plant appears healthy!**
- Continue current care routine
- Monitor regularly for changes
- Maintain proper watering and nutrition
- Ensure good air circulation
"""
                elif any(term in disease_lower for term in ['bacterial', 'spot', 'blight']):
                    fallback_text = """
**🦠 Bacterial/Fungal Disease Detected**

**Treatment:**
- Remove affected leaves immediately
- Apply copper-based fungicide
- Improve air circulation
- Avoid overhead watering

**Prevention:**
- Water at soil level
- Space plants properly
- Remove plant debris
- Rotate crops annually
"""
                elif any(term in disease_lower for term in ['rust', 'mildew', 'mold']):
                    fallback_text = """
**🍄 Fungal Disease Detected**

**Treatment:**
- Apply fungicidal spray
- Remove infected plant parts
- Increase air circulation
- Reduce humidity around plant

**Prevention:**
- Avoid watering leaves
- Ensure proper spacing
- Use disease-resistant varieties
- Apply preventive fungicide
"""
                elif 'virus' in disease_lower:
                    fallback_text = """
**🦠 Viral Disease Detected**

**Treatment:**
- Remove infected plants immediately
- Control insect vectors
- No chemical cure available
- Focus on preventing spread

**Prevention:**
- Use virus-free seeds
- Control aphids and other vectors
- Remove weeds that harbor viruses
- Quarantine new plants
"""
                else:
                    fallback_text = f"""
**🌿 Treatment for {predicted_disease}**

**General Treatment:**
- Remove affected plant parts
- Apply appropriate pesticide/fungicide
- Improve growing conditions
- Monitor plant closely

**Prevention:**
- Maintain plant health
- Ensure proper spacing and drainage
- Use disease-resistant varieties
- Practice crop rotation

**Consult:** Local agricultural extension service for specific treatment recommendations.
"""
                
                st.markdown(fallback_text)
                st.info("💡 **Tip:** Try restarting Ollama service for AI-powered detailed advice: `ollama serve`")
        
    else:
        st.error("Please upload an image to predict the disease.")

# Add footer with system info
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.info(f"🤖 AI Model: {OLLAMA_MODEL}")
with col2:
    st.info(f"🌐 Ollama Host: {OLLAMA_HOST}")