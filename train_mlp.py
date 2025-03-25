import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("IIT Tri Final SFRC.csv")  # Ensure the file is in your GitHub repo

# Define input (X) and target (y)
X = df.drop(columns=["Fs"])  # Assuming "Fs" is the target variable
y = df["Fs"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train MLP model
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp_model.fit(X_train_scaled, y_train)

# Save the MLP model and scaler
with open("mlp_model.pkl", "wb") as f:
    pickle.dump(mlp_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("MLP model and scaler saved successfully!")
