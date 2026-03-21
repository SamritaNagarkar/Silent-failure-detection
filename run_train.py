import pandas as pd
import joblib
import os
from src.train_model import train_model

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load datasets
X = pd.read_csv("data/X.csv")
y = pd.read_csv("data/y.csv")

# Convert y to series
y = y.squeeze()

print("Dataset loaded:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Train model
model = train_model(X, y)

# Save the trained model
model_path = "models/xgboost_model.pkl"
joblib.dump(model, model_path)
print(f"\nModel successfully saved to: {model_path}")