import pandas as pd
import glob
import os
from src.features import create_features

PROCESSED_DIR = "Data/processed/test"
LABEL_DIR = "Data/Raw/ServerMachineDataset/test_label"

all_data = []

csv_files = glob.glob(os.path.join(PROCESSED_DIR, "*.csv"))
csv_files.sort()

for csv_file in csv_files:
    machine_name = os.path.basename(csv_file).replace(".csv", "")
    label_file = os.path.join(LABEL_DIR, f"{machine_name}.txt")
    
    if not os.path.exists(label_file):
        print(f"Warning: Label file for {machine_name} not found at {label_file}, skipping...")
        continue
        
    print(f"Loading {machine_name}...")
    
    sensor_df = pd.read_csv(csv_file)
    label_df = pd.read_csv(label_file, header=None, names=["anomaly"])
    
    df = pd.concat([sensor_df, label_df], axis=1)
    df["machine_id"] = machine_name
    
    all_data.append(df)

full_df = pd.concat(all_data, ignore_index=True)
print("Combined dataset shape:", full_df.shape)

# Create features
full_df = create_features(full_df)

print("Feature dataset shape:", full_df.shape)

# Save feature dataset
full_df.to_csv("Data/feature_dataset.csv", index=False)
print("Feature dataset saved to Data/feature_dataset.csv")
