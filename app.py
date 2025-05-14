import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import io
from PIL import Image
import requests
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
from io import BytesIO
import threading
import time

# Load the model
model = load_model("PDDS.keras")

# Load class names
if os.path.exists("class_names.txt"):
    with open("class_names.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    class_names = [
        "Apple___Black_rot",
        "Apple___Scab",
        "Apple___healthy",
        "Corn___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn___Common_rust",
        "Corn___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___healthy",
        "Tomato___Early_blight"
    ]

# Create FastAPI app
app = FastAPI(title="Plant Disease Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preprocess image function
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# FastAPI endpoint for prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read and convert the file to PIL Image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Preprocess the image
    processed_img = preprocess_image(img)
    
    # Make prediction
    predictions = model.predict(processed_img)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx])
    
    # Return prediction
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "class_names": class_names,
        "probabilities": predictions[0].tolist()
    }

# Streamlit UI
def streamlit_app():
    # Set page config
    st.set_page_config(
        page_title="Plant Disease Detection",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for a fabulous UI
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: rgba(76, 175, 80, 0.1);
    }
    .result-box {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header {
        color: #2E7D32;
        text-align: center;
    }
    .subheader {
        color: #388E3C;
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #666;
        font-size: 12px;
        margin-top: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='header'>🌿 Plant Disease Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Upload a leaf image to detect diseases</h3>", unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display the uploaded image
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)
            
            # Add a predict button
            if st.button("Predict Disease", key="predict_button", help="Click to predict the disease"):
                with st.spinner("Analyzing leaf image..."):
                    # Convert image to bytes for API request
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_bytes = buffered.getvalue()
                    
                    # Make API request to FastAPI endpoint
                    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
                    response = requests.post("http://localhost:8000/predict/", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        predicted_class = result["predicted_class"]
                        confidence = result["confidence"] * 100
                        
                        # Display results in the second column
                        with col2:
                            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                            st.markdown("<h2 style='text-align: center; color: #2E7D32;'>Prediction Results</h2>", unsafe_allow_html=True)
                            
                            # Display the prediction
                            st.markdown(f"<h3 style='text-align: center;'>Detected: <span style='color: {'#D32F2F' if 'healthy' not in predicted_class else '#388E3C'};'>{predicted_class.replace('___', ' - ')}</span></h3>", unsafe_allow_html=True)
                            
                            # Display confidence
                            st.markdown(f"<h4 style='text-align: center;'>Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)
                            
                            # Display progress bar for confidence
                            st.progress(confidence/100)
                            
                            # Display recommendations based on disease
                            st.markdown("<h4>Recommendations:</h4>", unsafe_allow_html=True)
                            
                            if "healthy" in predicted_class:
                                st.success("Your plant appears to be healthy! Continue with regular care.")
                            else:
                                # Display disease-specific recommendations
                                if "Black_rot" in predicted_class:
                                    st.error("Black rot detected! Remove infected leaves and apply fungicide.")
                                    st.markdown("• Remove infected plant parts")
                                    st.markdown("• Apply copper-based fungicides")
                                    st.markdown("• Ensure good air circulation")
                                elif "Scab" in predicted_class:
                                    st.error("Scab detected! Apply fungicide and improve air circulation.")
                                    st.markdown("• Apply sulfur or copper-based fungicides")
                                    st.markdown("• Prune to improve air circulation")
                                    st.markdown("• Remove fallen leaves to reduce infection")
                                elif "Cercospora" in predicted_class or "Gray_leaf_spot" in predicted_class:
                                    st.error("Cercospora leaf spot detected! Apply fungicide and rotate crops.")
                                    st.markdown("• Apply fungicide treatments")
                                    st.markdown("• Practice crop rotation")
                                    st.markdown("• Improve air circulation")
                                elif "Common_rust" in predicted_class:
                                    st.error("Common rust detected! Apply fungicide and improve drainage.")
                                    st.markdown("• Apply fungicide treatments")
                                    st.markdown("• Improve soil drainage")
                                    st.markdown("• Remove infected plants")
                                elif "Esca" in predicted_class or "Black_Measles" in predicted_class:
                                    st.error("Esca (Black Measles) detected! Remove infected vines.")
                                    st.markdown("• Remove severely infected vines")
                                    st.markdown("• Apply fungicide preventatively")
                                    st.markdown("• Ensure proper pruning techniques")
                                elif "Early_blight" in predicted_class:
                                    st.error("Early blight detected! Remove infected leaves and apply fungicide.")
                                    st.markdown("• Remove infected leaves")
                                    st.markdown("• Apply fungicide treatments")
                                    st.markdown("• Mulch around plants")
                                else:
                                    st.error(f"Disease detected: {predicted_class}")
                                    st.markdown("• Consult with a plant pathologist")
                                    st.markdown("• Remove infected plant parts")
                                    st.markdown("• Consider appropriate fungicide treatments")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error("Error in prediction. Please try again.")
    
    # If no image is uploaded, show information in the second column
    if uploaded_file is None:
        with col2:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: #2E7D32;'>How It Works</h2>", unsafe_allow_html=True)
            st.markdown("""
            <ol>
                <li>Upload a clear image of a plant leaf</li>
                <li>Our AI model analyzes the image</li>
                <li>Get instant disease detection results</li>
                <li>Receive treatment recommendations</li>
            </ol>
            
            <h4>Supported Plants:</h4>
            <ul>
                <li>Apple</li>
                <li>Corn</li>
                <li>Grape</li>
                <li>Tomato</li>
            </ul>
            
            <h4>Detectable Diseases:</h4>
            <ul>
                <li>Black Rot</li>
                <li>Apple Scab</li>
                <li>Cercospora Leaf Spot</li>
                <li>Common Rust</li>
                <li>Esca (Black Measles)</li>
                <li>Early Blight</li>
                <li>And more...</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("<div class='footer'>Plant Disease Detection System © 2023</div>", unsafe_allow_html=True)

# Function to run FastAPI with uvicorn
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Main function to run both Streamlit and FastAPI
if __name__ == "__main__":
    # Start FastAPI in a separate thread
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    
    # Give FastAPI time to start
    time.sleep(2)
    
    # Run Streamlit app
    streamlit_app()
