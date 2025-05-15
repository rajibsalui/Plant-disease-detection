import streamlit as st
import tensorflow as tf
#from tf.keras.models import load_model
#from tensorflow.keras.preprocessing import image
import numpy as np
import os
import io
from PIL import Image
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
import threading
import time
import google.generativeai as genai
import dotenv

# Load the model
model = tf.keras.models.load_model("PDDS.keras")

# Load class names
if os.path.exists("class_names.txt"):
    with open("class_names.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    print("Error: class_names.txt file not found.")

# Load Gemini API key from .env file
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
    raise ValueError("GEMINI_API_KEY is required for Gemini API configuration.")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_recommendation(predicted_class, confidence):
    """
    Call Gemini API to get recommendations for the predicted disease.
    """
    prompt = f"""
    I have a plant leaf image that was classified as '{predicted_class.replace('___', ' - ')}' with a confidence of {confidence:.2f}%. 
    Please provide actionable, expert-level recommendations for what to do next, including treatment, prevention, and care tips. If the plant is healthy, provide tips to keep it healthy. Respond in markdown format.
    """
    
    try:
        # Use the correct model name for the current Gemini API version
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
        
        # Generate content with safety settings
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Return the generated text or a fallback message if empty
        if hasattr(response, 'text') and response.text:
            return response.text
        else:
            return "Sorry, I couldn't generate recommendations at this moment. Please try again."
            
    except Exception as e:
        # Provide a fallback response if the API call fails
        print(f"Gemini API Error: {str(e)}")
        return """
## Plant Care Recommendations

### General Care Tips
- Remove any visibly damaged or infected leaves
- Ensure proper watering (check soil moisture before watering)
- Provide adequate sunlight based on plant type
- Consider applying appropriate treatment based on the detected condition

*For detailed recommendations, please try again later when the AI service is available.*
"""

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
    img_array = tf.keras.preprocessing.image.img_to_array(img)
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
            st.image(img, caption="Uploaded Image", use_container_width=True)

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

                          

                            # Gemini AI Suggestions Section
                            st.markdown("""
                            <div class="gemini-container" style="background: linear-gradient(to right, #f8f9fa, #e9f7ef); border-radius: 12px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); border-left: 4px solid #34A853; margin-top: 25px;">
                                <div class="gemini-header" style="display: flex; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #e0e0e0; padding-bottom: 10px;">
                                    <img src="https://storage.googleapis.com/gweb-uniblog-publish-prod/images/gemini_advanced_logo.max-1000x1000.png" style="width: 32px; height: 32px; margin-right: 12px;">
                                    <div style="color: #1E3A8A; font-size: 20px; font-weight: 600;">Gemini AI Insights</div>
                                </div>
                            """, unsafe_allow_html=True)

                            # Create a container for Gemini insights
                            gemini_container = st.container()
                            with gemini_container:
                                # Add a loading spinner for Gemini
                                with st.spinner("Gemini AI is analyzing your plant..."):
                                    # Call Gemini API for recommendations
                                    gemini_recommendation = get_gemini_recommendation(predicted_class, confidence)
                                    
                                    # Add a quality indicator based on confidence
                                    quality_color = "#34A853" if confidence > 85 else "#FBBC05" if confidence > 70 else "#EA4335"
                                    quality_label = "High Confidence" if confidence > 85 else "Moderate Confidence" if confidence > 70 else "Low Confidence"
                                    
                                    st.markdown(f"""
                                    <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                                        <span style="background-color: {quality_color}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">
                                            {quality_label}
                                        </span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Display the Gemini recommendations in a styled container
                                    st.markdown(f"""
                                    <div style="background-color: white; border-radius: 8px; padding: 15px; box-shadow: inset 0 0 5px rgba(0,0,0,0.05);">
                                        {gemini_recommendation}
                                    </div>
                                    """, unsafe_allow_html=True)

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
    st.markdown("<div class='footer'>Plant Disease Detection System © 2025</div>", unsafe_allow_html=True)

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
