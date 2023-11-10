import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained SVM model
with open('svm_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load the pre-trained StandardScaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to preprocess user input and make predictions
def predict_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, density, pH, sulphates, alcohol, sulfur_dioxide_difference, type_white):
    input_data = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, density, pH, sulphates, alcohol, sulfur_dioxide_difference, type_white]).reshape(1, -1)
    scaled_input = scaler.transform(input_data)
    prediction = loaded_model.predict(scaled_input)
    return prediction

st.markdown(
    """
    <style>
    .st-ec, .st-ei, .st-eo, .st-el {
        background-color: yellow;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("Wine Quality Prediction App")
st.sidebar.header("Input Wine Parameters")

fixed_acidity = st.sidebar.slider("Fixed Acidity", min_value=0.0, max_value=15.0, value=7.0)
volatile_acidity = st.sidebar.slider("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.6)
citric_acid = st.sidebar.slider("Citric Acid", min_value=0.0, max_value=2.0, value=0.3)
residual_sugar = st.sidebar.slider("Residual Sugar", min_value=0.0, max_value=50.0, value=2.0)
chlorides = st.sidebar.slider("Chlorides", min_value=0.0, max_value=1.0, value=0.08)
density = st.sidebar.slider("Density", min_value=0.9, max_value=1.1, value=0.997)
pH = st.sidebar.slider("pH", min_value=2.0, max_value=5.0, value=3.0)
sulphates = st.sidebar.slider("Sulphates", min_value=0.0, max_value=2.0, value=0.5)
alcohol = st.sidebar.slider("Alcohol", min_value=0.0, max_value=20.0, value=10.0)
sulfur_dioxide_difference = st.sidebar.slider("Sulfur Dioxide Difference", min_value=0.0, max_value=200.0, value=0.0)
type_white = st.sidebar.selectbox("Wine Type", ["Red", "White"])

if st.button("Predict"):
    prediction = predict_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, density, pH, sulphates, alcohol, sulfur_dioxide_difference, 1 if type_white == 'White' else 0)
    if prediction[0] == 1:
        st.error("You better have a beer, this wine is a Fraud!!!")
        st.image("1573393-1608114081883_Presentational-Promotional-Image-1080x1080.jpeg", use_column_width=True)
    else:
        if type_white == "White":
            st.image("What-Is-Muscat-Wine-FT-BLOG0922-2000-2b4cc742956242c994174021cf5eaf68.jpeg", use_column_width=True)
            st.markdown("This white wine is legit and ready for consumption!!!")
        else:
            st.image("curso-cata-madrid.jpeg", use_column_width=True)
            st.markdown("This red wine is legit and ready for consumption!!!")

st.sidebar.text("Make predictions with user inputs.")



