import pandas as pd


def create_early_warning_label(df, window=30):
    """
    Create early warning labels.
    Marks rows before an anomaly as potential failures.
    Operates per machine_id to avoid leakage.
    """

    df["early_warning"] = 0

    if "machine_id" in df.columns:
        # We need to process each machine independently so we don't bleed labels across machines
        for machine, group in df.groupby("machine_id"):
            anomaly_indices = group.index[group["anomaly"] == 1]
            for idx in anomaly_indices:
                # Find the start index. It must not go below the first index of this group
                start = max(group.index.min(), idx - window)
                df.loc[start:idx, "early_warning"] = 1
    else:
        # Fallback for single machine
        anomaly_indices = df.index[df["anomaly"] == 1]
        for idx in anomaly_indices:
            start = max(0, idx - window)
            df.loc[start:idx, "early_warning"] = 1

    return df


def build_dataset(df):
    """
    Build ML dataset with early warning labels
    """

    df = create_early_warning_label(df)

    y = df["early_warning"]
    
    # Drop labels that shouldn't be used as features
    cols_to_drop = ["anomaly", "early_warning"]
    # We DO NOT drop machine_id here anymore, so we can use it for GroupShuffleSplit in train_model.py
        
    X = df.drop(columns=cols_to_drop)

    print("Feature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)

    return X, y
