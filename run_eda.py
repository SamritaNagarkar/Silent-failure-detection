from src.eda import load_machine_data
from src.eda import plot_anomaly_timeline
from src.eda import plot_sensor_trends
from src.eda import plot_failure_window

DATA_PATH = "data/processed/test/machine-1-1.csv"
LABEL_PATH = "data/raw/ServerMachineDataset/test_label/machine-1-1.txt"

# Load data
df = load_machine_data(DATA_PATH, LABEL_PATH)

print("Dataset loaded:", df.shape)

# 1️⃣ Plot anomaly timeline
plot_anomaly_timeline(df)

# 2️⃣ Plot some sensor trends
sensors = ["sensor_1", "sensor_2", "sensor_3"]
plot_sensor_trends(df, sensors, start=0, end=2000)

# 3️⃣ Plot sensor behavior before failure
plot_failure_window(df, sensors, window=300)