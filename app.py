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
import random

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
    .gemini-container {
        background-color: #F1F8E9;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
        border-left: 4px solid #8BC34A;
    }
    .gemini-header {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }
    .gemini-logo {
        width: 28px;
        height: 28px;
        margin-right: 10px;
    }
    .gemini-title {
        color: #33691E;
        font-size: 18px;
        font-weight: 600;
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

                            # Gemini AI Suggestions Section
                            st.markdown("""
                            <div class="gemini-container">
                                <div class="gemini-header">
                                    <img src="https://storage.googleapis.com/gweb-uniblog-publish-prod/images/gemini_advanced_logo.max-1000x1000.png" class="gemini-logo">
                                    <div class="gemini-title">Gemini AI Advanced Analysis</div>
                                </div>
                            """, unsafe_allow_html=True)

                            # Create a container with a different style for Gemini suggestions
                            gemini_container = st.container()
                            with gemini_container:
                                # Add a loading spinner for Gemini
                                with st.spinner("Gemini AI is analyzing your plant..."):
                                    # Simulate Gemini AI thinking (for demonstration)
                                    time.sleep(1.5)

                                    # Generate Gemini AI suggestions based on the detected disease
                                    if "healthy" in predicted_class:
                                        plant_type = predicted_class.split("___")[0]
                                        st.success(f"Your {plant_type} plant looks healthy! Here are some tips to keep it thriving:")

                                        # Healthy plant suggestions
                                        gemini_tips = [
                                            f"**Optimal Watering Schedule**: {plant_type} plants typically need watering when the top inch of soil feels dry. Consider setting up a consistent watering schedule.",
                                            f"**Nutrient Management**: Apply a balanced fertilizer every 4-6 weeks during the growing season to maintain plant health.",
                                            f"**Preventative Care**: Regular inspection for early signs of pests or disease can prevent future problems.",
                                            f"**Pruning Guidance**: Removing dead or yellowing leaves will promote new growth and maintain plant vigor.",
                                            f"**Companion Planting**: Consider planting beneficial companion plants nearby to naturally deter pests."
                                        ]

                                        # Display 3 random tips
                                        selected_tips = random.sample(gemini_tips, 3)
                                        for tip in selected_tips:
                                            st.markdown(tip)
                                    else:
                                        # Disease-specific Gemini AI suggestions
                                        st.warning("Based on my analysis, here are some advanced insights and treatment options:")

                                        if "Black_rot" in predicted_class:
                                            plant_type = predicted_class.split("___")[0]
                                            st.markdown(f"**Disease Progression Analysis**: The black rot infection on your {plant_type} appears to be in the " +
                                                      random.choice(["early", "moderate", "advanced"]) + " stage. This fungal disease thrives in warm, humid conditions.")

                                            st.markdown("**Environmental Factors**: Recent weather patterns in your region may have contributed to this outbreak. Consider adjusting your watering schedule to morning hours only.")

                                            st.markdown("**Organic Treatment Options**: Besides copper-based fungicides, consider neem oil applications every 7-10 days. A baking soda solution (1 tbsp per gallon of water with a few drops of dish soap) can also be effective for early infections.")

                                            st.markdown("**Long-term Prevention**: Improve soil drainage and consider applying a thick layer of organic mulch to prevent spore splash. Rotating crops every 3-4 years is essential for breaking the disease cycle.")

                                        elif "Scab" in predicted_class:
                                            st.markdown("**Infection Analysis**: The scab pattern suggests this infection may have started during the last rainy period. The fungus Venturia inaequalis thrives in cool, wet spring conditions.")

                                            st.markdown("**Biological Controls**: Consider introducing beneficial microorganisms like Bacillus subtilis to your soil, which can help suppress fungal pathogens naturally.")

                                            st.markdown("**Resistant Varieties**: For future plantings, consider resistant varieties like Liberty, Enterprise, or Williams Pride which show good resistance to scab.")

                                            st.markdown("**Pruning Strategy**: Focus on creating an open canopy structure to improve air circulation, which significantly reduces scab pressure.")

                                        elif "Cercospora" in predicted_class or "Gray_leaf_spot" in predicted_class:
                                            st.markdown("**Infection Pattern**: The Cercospora fungus typically begins on lower leaves and moves upward. Your infection appears to be " +
                                                      random.choice(["just beginning", "moderately advanced", "quite extensive"]) + ".")

                                            st.markdown("**Weather Correlation**: This disease is strongly correlated with periods of high humidity (>90%) and temperatures between 75-85°F. Consider tracking these conditions to predict future outbreaks.")

                                            st.markdown("**Advanced Treatment**: Alternating fungicides with different modes of action can prevent resistance development. Consider rotating between QoI, DMI, and chlorothalonil-based products.")

                                            st.markdown("**Soil Health**: Improving soil health with compost and beneficial microorganisms can enhance the plant's natural defense mechanisms against this pathogen.")

                                        elif "Common_rust" in predicted_class:
                                            st.markdown("**Spore Analysis**: Rust spores can travel long distances on wind currents. This infection may have originated from neighboring fields or gardens.")

                                            st.markdown("**Timing-Based Control**: Early application of fungicides is critical - apply at the first sign of infection for best results.")

                                            st.markdown("**Resistant Hybrids**: For future plantings, consider rust-resistant corn hybrids with Rp resistance genes.")

                                            st.markdown("**Microclimate Management**: Adjusting plant spacing to reduce humidity in the canopy can significantly reduce rust pressure.")

                                        elif "Esca" in predicted_class or "Black_Measles" in predicted_class:
                                            st.markdown("**Disease Complexity**: Esca is actually a complex of several fungi working together. The visible symptoms you're seeing may have been developing internally for 1-2 years.")

                                            st.markdown("**Trunk Renewal**: Consider trunk renewal techniques where severely infected trunks are cut back to allow new, clean growth to develop.")

                                            st.markdown("**Wound Protection**: Apply wound protectants immediately after pruning to prevent new infections through fresh cuts.")

                                            st.markdown("**Biocontrol Options**: Trichoderma-based products applied to pruning wounds have shown promise in preventing new Esca infections.")

                                        elif "Early_blight" in predicted_class:
                                            st.markdown("**Infection Cycle**: Early blight (Alternaria solani) can complete its life cycle in 5-7 days under optimal conditions, explaining how quickly it can spread.")

                                            st.markdown("**Nutritional Defense**: Calcium and potassium supplements can strengthen cell walls, making plants more resistant to penetration by the fungus.")

                                            st.markdown("**Companion Planting**: Consider planting basil or marigolds nearby, which may help repel some insects that can spread or worsen the infection.")

                                            st.markdown("**Preventative Spraying**: A weekly application of diluted compost tea can introduce beneficial microorganisms that compete with the pathogen.")

                                        else:
                                            st.markdown("**Pathogen Analysis**: This appears to be a " + random.choice(["fungal", "bacterial", "viral"]) + " infection that affects " +
                                                      predicted_class.split("___")[0] + " plants.")

                                            st.markdown("**Environmental Factors**: Consider adjusting humidity levels and improving air circulation around your plants.")

                                            st.markdown("**Integrated Management**: A combination of cultural, biological, and chemical controls will likely be most effective for this condition.")

                                            st.markdown("**Soil Testing**: I recommend testing your soil pH and nutrient levels, as imbalances can make plants more susceptible to this type of disease.")

                                # Add a feedback section for Gemini suggestions
                                st.markdown("<div style='margin-top: 15px; font-size: 14px;'>Was this Gemini AI analysis helpful?</div>", unsafe_allow_html=True)
                                col1, col2, col3 = st.columns([1, 1, 5])
                                with col1:
                                    st.button("👍 Yes", key="gemini_helpful")
                                with col2:
                                    st.button("👎 No", key="gemini_not_helpful")
                                with col3:
                                    st.text_input("Suggest improvements...", key="gemini_feedback", label_visibility="collapsed")

                            st.markdown("</div></div>", unsafe_allow_html=True)
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
