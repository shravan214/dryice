import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load ML models
with open("mlp_model.pkl", "rb") as f:
    mlp_model = pickle.load(f)
with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)
with open("linear_reg_model.pkl", "rb") as f:
    linear_reg_model = pickle.load(f)

# Title
st.title('Reinforced Concrete')

# Section 1: Database Preparation
st.header("1. Database Preparation")
if st.checkbox("1.1 Show database of the flexural strength of the SFRC"):
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
if st.checkbox("1.2 Show data distribution"):
    if uploaded_file is not None:
        st.write("### Data Distribution")
        plt.figure(figsize=(10, 6))
        sns.histplot(df["Fs"], bins=20, kde=True)
        st.pyplot(plt)

# Section 2: Machine Learning Approaches
st.header("2. Machine Learning Approaches")
if st.checkbox("2.1 Show structure of k-Nearest Neighbor model"):
    st.write(knn_model)
if st.checkbox("2.2 Show structure of Multiple Linear Regression model"):
    st.write(linear_reg_model)

# Section 3: Predicting Flexural Strength
st.header("3. Predicting Flexural Strength of Steel Fiber-Reinforced Concrete")

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
    prediction = mlp_model.predict(input_data)[0]
    st.write(f"### Predicted Flexural Strength: {prediction:.2f} MPa")

