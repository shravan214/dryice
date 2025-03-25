import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load MLP model and scaler
with open("train_mlp.pkl", "rb") as f:
    train_mlp = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Title
st.title('Predicting Flexural Strength of SFRC')

# Input sliders
cement = st.slider("Cement (kg/m³)", 300, 700, 400)
water = st.slider("Water (kg/m³)", 100, 250, 150)
superplasticizer = st.slider("Superplasticizer (%)", 0.0, 2.0, 0.5)
silica_fume = st.slider("Silica fume (kg/m³)", 0, 50, 10)
aspect_ratio = st.slider("Aspect ratio of fiber", 0, 100, 50)
fiber_volume = st.slider("Fiber volume fraction (%)", 0.0, 5.0, 1.0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[cement, water, superplasticizer, silica_fume, aspect_ratio, fiber_volume]])
    input_data_scaled = scaler.transform(input_data)  # Scale input before prediction
    prediction = train_mlp.predict(input_data_scaled)[0]
    st.write(f"### Predicted Flexural Strength: {prediction:.2f} MPa")
