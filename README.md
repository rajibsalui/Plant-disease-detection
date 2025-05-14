# Plant Disease Detection System

A web application for detecting plant leaf diseases using machine learning. This application combines Streamlit for the frontend UI and FastAPI for the backend API, all running with uvicorn.

## Features

- 🌿 Upload leaf images to detect diseases
- 🔍 AI-powered disease detection
- 📊 Confidence scores for predictions
- 💡 Treatment recommendations based on detected diseases
- 🎨 Beautiful and intuitive user interface

## Supported Plants and Diseases

- **Apple**: Black Rot, Scab, Healthy
- **Corn**: Cercospora Leaf Spot (Gray Leaf Spot), Common Rust, Healthy
- **Grape**: Black Rot, Esca (Black Measles), Healthy
- **Tomato**: Early Blight, and more

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/plant-disease-detection.git
   cd plant-disease-detection
   ```

2. Install the required dependencies:
   ```
   pip install streamlit fastapi uvicorn tensorflow pillow python-multipart
   ```

3. Create the model (if not already created):
   ```
   python create_model.py
   ```

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   - Streamlit UI: http://localhost:8501
   - FastAPI Documentation: http://localhost:8000/docs

3. Upload a leaf image and click "Predict Disease" to get results.

## How It Works

1. The application uses a convolutional neural network (CNN) trained on a dataset of plant leaf images.
2. When you upload an image, it's sent to the FastAPI backend for processing.
3. The image is preprocessed and fed into the model for prediction.
4. Results are displayed in the Streamlit UI with confidence scores and recommendations.

## Project Structure

- `app.py`: Main application file containing both Streamlit UI and FastAPI backend
- `create_model.py`: Script to create and save the model
- `PDDS.keras`: Trained model file
- `class_names.txt`: List of class names for prediction

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Streamlit
- FastAPI
- Uvicorn
- Pillow
- Python-multipart

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Plant Village Dataset
- TensorFlow and Keras
- Streamlit and FastAPI communities
