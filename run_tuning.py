import pandas as pd
from src.tune_model import tune_xgboost

print("Loading dataset...")
X = pd.read_csv("data/X.csv")
y = pd.read_csv("data/y.csv").squeeze()

# Run tuning process
# We'll run 5 iterations just to demonstrate. In a real scenario, this could be 20-50.
best_model, best_params = tune_xgboost(X, y, n_iter=5)

print("\nTo use these parameters, update src/train_model.py with the new values.")
