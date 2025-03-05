import streamlit as st
import pickle
import numpy as np
import os

# Cache model loading for performance optimization
@st.cache_resource
def load_model():
    # Get absolute path of current directory
    base_path = os.path.dirname(__file__)

    # Construct full paths for model and label encoder
    model_path = os.path.join(base_path, "model.pkl")
    encoder_path = os.path.join(base_path, "label_encoder.pkl")

    # Load model and label encoder safely
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        return model, encoder
    except FileNotFoundError:
        st.error("Model or label encoder file not found. Please check your repository.")
        return None, None


def main():
    st.set_page_config(page_title="Calorie Prediction App", layout="centered")
    st.title("🔥 Calorie Prediction App")

    # Load the model and encoder
    model, encoder = load_model()

    if model is None or encoder is None:
        st.error("Failed to load model or encoder. Please check deployment.")
        return

    label_classes = encoder.classes_

    # Input fields
    age = st.number_input("Enter Age:", min_value=0, max_value=100, step=1)
    height = st.number_input("Enter Height (cm):", min_value=50.0, max_value=250.0, step=0.1)
    weight = st.number_input("Enter Weight (kg):", min_value=20.0, max_value=200.0, step=0.1)
    gender = st.selectbox("Select Gender", ["Male", "Female"])

    # Encoding gender
    gender_encoded = 1 if gender == "Male" else 0

    # Predict Button
    if st.button("Predict Calories"):
        input_data = np.array([[age, height, weight, gender_encoded]])
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Daily Calorie Intake: {prediction:.2f} kcal")


if __name__ == "__main__":
    main()
