import os
import streamlit as st
import pickle
import numpy as np

# Load Model Function
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl") or not os.path.exists("label_encoder.pkl"):
        st.error("⚠️ Model files not found! Upload `model.pkl` and `label_encoder.pkl` to your repository.")
        return None, None

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, encoder

# Streamlit UI
def main():
    st.set_page_config(page_title="Calorie Prediction App", layout="centered")
    st.title("🍽️ Calorie Prediction App")

    model, encoder = load_model()

    if model is None or encoder is None:
        return

    label_classes = encoder.classes_

    # Custom CSS Styling
    st.markdown("""
    <style>
        body {
            background-image: url('https://img.freepik.com/free-photo/top-view-bowls-with-veggies-fruit-copy-space_23-2148585684.jpg');
            background-size: cover;
            font-family: Arial, sans-serif;
        }
        .stApp {
            background: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #ff9800;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #e68900;
        }
    </style>
    """, unsafe_allow_html=True)

    # Layout with two columns
    col1, col2 = st.columns(2)
    with col1:
        label = st.selectbox("Select Food Label", label_classes)
        weight = st.number_input("Weight (g)", min_value=10, max_value=1000, value=100)
        protein = st.number_input("Protein (g)", min_value=0, max_value=100, value=10)
        carbohydrates = st.number_input("Carbohydrates (g)", min_value=0, max_value=200, value=20)

    with col2:
        fats = st.number_input("Fats (g)", min_value=0, max_value=100, value=5)
        fiber = st.number_input("Fiber (g)", min_value=0, max_value=100, value=2)
        sugars = st.number_input("Sugars (g)", min_value=0, max_value=100, value=5)
        sodium = st.number_input("Sodium (mg)", min_value=0, max_value=5000, value=10)

    encoded_label = encoder.transform([label])[0]
    input_data = np.array([encoded_label, weight, protein, carbohydrates, fats, fiber, sugars, sodium]).reshape(1, -1)

    if st.button("Predict Calories"):
        predicted_calories = model.predict(input_data)[0]
        st.success(f"🔥 Predicted Calories: **{predicted_calories:.2f} kcal**")

if __name__ == "__main__":
    main()
