import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, average_precision_score

def tune_xgboost(X, y, n_iter=5):
    """
    Perform hyperparameter tuning for XGBoost using TimeSeriesSplit.
    """
    print("Starting hyperparameter tuning...")
    print(f"Dataset size: {X.shape}")

    # Replace inf values with nan
    X = X.copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Calculate scale_pos_weight
    neg_count = np.sum(y == 0)
    pos_count = np.sum(y == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Parameter grid for XGBoost
    param_dist = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 1, 5]
    }

    # Initialize base model
    xgb_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr'
    )

    # TimeSeriesSplit prevents data leakage from future to past
    tscv = TimeSeriesSplit(n_splits=3)

    # Use Average Precision (PR-AUC) as the optimization metric
    scorer = make_scorer(average_precision_score, response_method='predict_proba')

    # Setup RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scorer,
        cv=tscv,
        verbose=2,
        random_state=42,
        n_jobs=1  # XGBoost already uses all cores, so n_jobs=1 for the search
    )

    print(f"\nRunning RandomizedSearchCV with {n_iter} iterations and 3 folds...")
    random_search.fit(X, y)

    print("\n--- Tuning Completed ---")
    print(f"Best PR-AUC Score: {random_search.best_score_:.4f}")
    print("Best Parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")

    return random_search.best_estimator_, random_search.best_params_
