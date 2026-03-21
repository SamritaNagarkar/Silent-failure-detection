import pandas as pd
import numpy as np


def vectorized_rolling_slope(series, window):
    """
    Fast, vectorized calculation of rolling slope using linear regression.
    """
    x = np.arange(window)
    x_mean = x.mean()
    # Sum of squared deviations for x
    x_var_sum = np.sum((x - x_mean)**2)
    
    # Precompute weights for the dot product
    weights = (x - x_mean) / x_var_sum

    # We can use rolling apply with raw=True for fast numpy array operations
    def slope_func(y):
        return np.dot(weights, y)
        
    return series.rolling(window=window).apply(slope_func, raw=True)


def create_features(df, machine_id_col="machine_id"):
    """
    Create rolling and lag features. 
    If machine_id_col is provided, computes features per machine to avoid 
    data leakage between different machines.
    """
    sensor_cols = ["sensor_1", "sensor_2", "sensor_3"]
    
    if machine_id_col in df.columns:
        print(f"Creating features grouped by {machine_id_col}...")
        grouper = df.groupby(machine_id_col)
    else:
        print("Creating features for a single continuous dataset...")
        grouper = df  # fallback if no machine_id is provided

    for sensor in sensor_cols:

        # Lag features
        df[f"{sensor}_lag_1"] = grouper[sensor].shift(1)
        df[f"{sensor}_lag_5"] = grouper[sensor].shift(5)
        df[f"{sensor}_lag_10"] = grouper[sensor].shift(10)

        # Rolling statistics
        df[f"{sensor}_mean_50"] = grouper[sensor].transform(lambda x: x.rolling(window=50).mean())
        df[f"{sensor}_std_50"] = grouper[sensor].transform(lambda x: x.rolling(window=50).std())
        df[f"{sensor}_min_50"] = grouper[sensor].transform(lambda x: x.rolling(window=50).min())
        df[f"{sensor}_max_50"] = grouper[sensor].transform(lambda x: x.rolling(window=50).max())

        # Change features
        df[f"{sensor}_diff"] = grouper[sensor].diff()
        df[f"{sensor}_pct_change"] = grouper[sensor].pct_change()

        # Rolling slope (trend)
        df[f"{sensor}_slope_50"] = grouper[sensor].transform(lambda x: vectorized_rolling_slope(x, 50))

        # Simple anomaly score
        df[f"{sensor}_anomaly"] = (
            df[sensor] - df[f"{sensor}_mean_50"]
        ) / df[f"{sensor}_std_50"]

    return df
