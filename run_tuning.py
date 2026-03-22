import pandas as pd
from src.tune_model import tune_xgboost

print("Loading dataset...")
X = pd.read_csv("data/X.csv")
y = pd.read_csv("data/y.csv").squeeze()

# Run tuning process
# Running 15 iterations to search a wider space
best_model, best_params = tune_xgboost(X, y, n_iter=15)

print("\nTo use these parameters, update src/train_model.py with the new values.")
