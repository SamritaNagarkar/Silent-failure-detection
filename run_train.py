import pandas as pd
from src.train_model import train_model

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