import streamlit as st
import pickle
import numpy as np

# Cache model loading for performance optimization
@st.cache_resource
def load_model():
    model = pickle.load(open('model.pkl', 'rb'))
    encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    return model, encoder

def main():
    st.set_page_config(page_title="Calorie Prediction App", layout="centered")
    st.title("🍽️ Calorie Prediction App")
    
    model, encoder = load_model()
    label_classes = encoder.classes_
    
    st.markdown("""
    <style>
        body {
            background-image: url('https://img.freepik.com/free-photo/top-view-bowls-with-veggies-fruit-copy-space_23-2148585684.jpg?t=st=1741158623~exp=1741162223~hmac=8d0a0ca18d99d8c3a381027bf2493de9c1e100b293a241b98298d034ed55fa41&w=1060');
            background-size:cover;
            font-family: Arial, sans-serif;
        }
        .stApp {
            background: rgba(0.2, 5.0, 0.5,0.2);
            padding: 10px;
            border-radius: 10px;
            color: white;
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
    
    # Layout with two columns for better UX
    col1, col2 = st.columns(2)
    with col1:
        label = st.selectbox("Select Food Label", label_classes)
        weight = st.number_input("Weight (g)", min_value=18, max_value=1000, value=30)
        protein = st.number_input("Protein (g)", min_value=0, max_value=1000, value=10)
        carbohydrates = st.number_input("Carbohydrates (g)", min_value=0, max_value=1000, value=20)
    
    with col2:
        fats = st.number_input("Fats (g)", min_value=0, max_value=1000, value=5)
        fiber = st.number_input("Fiber (g)", min_value=0, max_value=1000, value=2)
        sugars = st.number_input("Sugars (g)", min_value=0, max_value=1000, value=5)
        sodium = st.number_input("Sodium (mg)", min_value=10, max_value=1000, value=10)
    
    encoded_label = encoder.transform([label])[0]
    input_data = np.array([encoded_label, weight, protein, carbohydrates, fats, fiber, sugars, sodium]).reshape(1, -1)
    
    if st.button("Predict Calories"):
        predicted_calories = model.predict(input_data)[0]
        st.success(f"Predicted Calories: **{predicted_calories:.2f} kcal**")

if __name__ == "__main__":
    main()
