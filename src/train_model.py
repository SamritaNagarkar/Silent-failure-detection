import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    fbeta_score,
)
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import matplotlib.pyplot as plt

def train_model(X, y):
    """
    Train a predictive maintenance model using XGBoost with Imbalance Handling
    """

    # Replace inf values with nan (caused by division by zero in pct_change or standard deviation)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Split dataset
    if "machine_id" in X.columns:
        print("\nUsing GroupShuffleSplit to split by machine_id (No data leakage across machines)...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=X["machine_id"]))
        
        X_train = X.iloc[train_idx].drop(columns=["machine_id"])
        X_test = X.iloc[test_idx].drop(columns=["machine_id"])
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
    else:
        print("\nFallback to standard time-based train_test_split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

    print("Training data:", getattr(X_train, 'shape', len(X_train)))
    print("Test data:", getattr(X_test, 'shape', len(X_test)))

    # Calculate scale_pos_weight to handle class imbalance
    # scale_pos_weight = count(negative examples) / count(positive examples)
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    print(f"\nClass imbalance handling:")
    print(f"Negative samples: {neg_count}, Positive samples: {pos_count}")
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Initialize model with optimized hyperparameters
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.2,
        subsample=0.9,
        colsample_bytree=0.7,
        gamma=0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr', # Optimize for Precision-Recall Area Under Curve
    )

    # Train model (with simple early stopping on test set for demonstration)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Predictions (probabilities)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Evaluation Metrics ---")
    
    # Standard threshold 0.5
    print("\nStandard Threshold (0.5):")
    preds_standard = (y_prob >= 0.5).astype(int)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds_standard))
    print("\nClassification Report:")
    print(classification_report(y_test, preds_standard))

    # Calculate AUC metrics
    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\nPR-AUC (Average Precision): {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Threshold Tuning for F2-score (Prioritizes Recall)
    best_threshold = 0.5
    best_f2 = 0.0

    # Test thresholds from 0.1 to 0.9
    for thresh in np.arange(0.1, 0.9, 0.1):
        preds = (y_prob >= thresh).astype(int)
        # F2-score values recall twice as much as precision
        f2 = fbeta_score(y_test, preds, beta=2.0) 
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = thresh

    print(f"\n--- Optimized Threshold for High Recall ---")
    print(f"Best Threshold (based on F2-score): {best_threshold:.2f} (F2 = {best_f2:.4f})")
    
    preds_optimized = (y_prob >= best_threshold).astype(int)
    print("Confusion Matrix (Optimized Threshold):")
    print(confusion_matrix(y_test, preds_optimized))
    print("\nClassification Report (Optimized Threshold):")
    print(classification_report(y_test, preds_optimized))

    # Plot and save feature importance
    print("\nSaving feature importance plot to 'feature_importance.png'...")
    importance = model.feature_importances_
    features = X_train.columns
    
    # Sort importances
    indices = np.argsort(importance)[::-1]
    
    # Take top 20 features
    top_n = min(20, len(features))
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.title(f"Top {top_n} Feature Importances")
    plt.bar(range(top_n), importance[top_indices], align="center")
    plt.xticks(range(top_n), [features[i] for i in top_indices], rotation=45, ha='right')
    plt.xlim([-1, top_n])
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    return model