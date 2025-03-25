import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("SFRCdata.csv")  # Ensure the file is in your GitHub repo

# Define input (X) and target (y)
X = df.drop(columns=["Flexural Strength"])  # Assuming "Fs" is the target variable
y = df["Flexural Strength"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train MLP model
train_mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
train_mlp.fit(X_train_scaled, y_train)

# Save the MLP model and scaler
with open("train_mlp.pkl", "wb") as f:
    pickle.dump(train_mlp, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("MLP model and scaler saved successfully!")
