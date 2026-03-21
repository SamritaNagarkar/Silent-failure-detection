import pandas as pd
from src.dataset import build_dataset

# Load feature dataset
df = pd.read_csv("data/feature_dataset.csv")

print("Loaded dataset:", df.shape)

# Build ML dataset
X, y = build_dataset(df)

# Save datasets
X.to_csv("data/X.csv", index=False)
y.to_csv("data/y.csv", index=False)

print("Saved:")
print("X = data/X.csv")
print("y = data/y.csv")