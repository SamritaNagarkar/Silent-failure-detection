import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path("Data/processed/test")
LABEL_DIR = Path("Data/Raw/ServerMachineDataset/test_label")

# Start with ONE machine
MACHINE_NAME = "machine-1-1"

data_file = DATA_DIR / f"{MACHINE_NAME}.csv"
label_file = LABEL_DIR / f"{MACHINE_NAME}.txt"

print("Loading sensor data from:", data_file)
print("Loading labels from:", label_file)

# Load sensor data
sensor_df = pd.read_csv(data_file)

# Load anomaly labels
label_df = pd.read_csv(label_file, header=None, names=["anomaly"])

# Sanity check
print("Sensor rows:", sensor_df.shape[0])
print("Label rows:", label_df.shape[0])

# Combine
sensor_df["anomaly"] = label_df["anomaly"]

print("\nCombined data loaded successfully")
print("Shape:", sensor_df.shape)
print("\nFirst 5 rows:")
print(sensor_df.head())

print("\nAnomaly distribution:")
print(sensor_df["anomaly"].value_counts())