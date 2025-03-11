import streamlit as st
import joblib
import numpy as np
import base64

# Load trained model
model = joblib.load("rf_model.pkl")

# Set Page Config
st.set_page_config(page_title="Breast Cancer Detection", page_icon="🔬", layout="centered")

# Function to set background image using base64 encoding
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{base64_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image filename
set_background("premium_photo-1681398718759-f77c1d059141.jpg")

# Custom CSS for Styling
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        html, body, [class*="st-"] {
            font-family: 'Poppins', sans-serif;
            color: black;
        }

        .title { text-align: center; font-size: 36px; font-weight: 600; color: black; }
        
        .subheader { text-align: center; font-size: 20px; color: black; font-weight: 400; }

        /* Modal (center pop-up) */
        .modal-container {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 40%;
            background-color: white;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            animation: fadeIn 0.3s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translate(-50%, -55%); }
            to { opacity: 1; transform: translate(-50%, -50%); }
        }

    </style>
    """, unsafe_allow_html=True
)

# UI Title
st.markdown("<h1 class='title'>🔬 Breast Cancer Detection</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subheader'>Enter tumor features below to predict.</h3>", unsafe_allow_html=True)
st.write("")

# Define the top 10 important features
features = [
    "area_worst", "concave points_worst", "radius_worst", "perimeter_worst",
    "concave points_mean", "perimeter_mean", "radius_mean", "concavity_mean",
    "area_mean", "concavity_worst"
]

# Create a two-column layout for inputs
col1, col2 = st.columns(2)
user_input = []

for i, feature in enumerate(features):
    value = (col1 if i % 2 == 0 else col2).number_input(f"**{feature.replace('_', ' ').title()}**", value=0.0, format="%.5f")
    user_input.append(value)

st.write("")  # Add spacing

# Predict button
if st.button("🔍 Predict", use_container_width=True):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]

    # Define the message and color based on prediction
    if prediction == 1:
        message = "⚠️ Malignant Tumor Detected! Consult a doctor immediately."
        bg_color = "#ff6b6b"
    else:
        message = "✅ Benign Tumor Detected! No signs of cancer."
        bg_color = "#2ecc71"

    # Display center pop-up modal
    st.markdown(
        f"""
        <div class="modal-container" style="background-color: {bg_color}; color: black;">
            <h2>{message}</h2>
        </div>
        """, unsafe_allow_html=True
    )
