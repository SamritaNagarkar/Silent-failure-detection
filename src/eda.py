import pandas as pd
import matplotlib.pyplot as plt

# Load combined data
DATA_PATH = "Data/processed/test/machine-1-1.csv"
LABEL_PATH = "Data/Raw/ServerMachineDataset/test_label/machine-1-1.txt"

# Load sensor data
sensor_df = pd.read_csv(DATA_PATH)

# Load labels
label_df = pd.read_csv(LABEL_PATH, header=None, names=["anomaly"])

# Combine
df = pd.concat([sensor_df, label_df], axis=1)

print("Dataset shape:", df.shape)
print(df["anomaly"].value_counts())

# Plot anomaly over time
plt.figure(figsize=(12, 3))
plt.plot(df.index, df["anomaly"], color="red", linewidth=0.8)
plt.title("Anomalies Over Time")
plt.xlabel("Time Index")
plt.ylabel("Anomaly (0 = Normal, 1 = Failure)")
plt.tight_layout()
plt.show()