import pandas as pd
import matplotlib.pyplot as plt


def load_machine_data(data_path, label_path):
    """
    Load sensor data and anomaly labels
    """
    sensor_df = pd.read_csv(data_path)
    label_df = pd.read_csv(label_path, header=None, names=["anomaly"])

    df = pd.concat([sensor_df, label_df], axis=1)
    return df


def plot_anomaly_timeline(df):
    """
    Plot anomaly labels over time
    """
    plt.figure(figsize=(12, 3))
    plt.plot(df.index, df["anomaly"], linewidth=0.8)
    plt.title("Anomaly Timeline")
    plt.xlabel("Time")
    plt.ylabel("Anomaly")
    plt.tight_layout()
    plt.show()


def plot_sensor_trends(df, sensors, start=0, end=2000):
    """
    Plot selected sensors in a given time range
    """
    subset = df.iloc[start:end]

    plt.figure(figsize=(12, 6))

    for sensor in sensors:
        plt.plot(subset.index, subset[sensor], label=sensor)

    plt.title("Sensor Trends")
    plt.xlabel("Time")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_failure_window(df, sensors, window=300):
    """
    Plot sensor behavior before failure events
    """

    failure_indices = df[df["anomaly"] == 1].index

    for idx in failure_indices[:3]:   # show first few failures
        start = max(0, idx - window)
        subset = df.iloc[start:idx]

        plt.figure(figsize=(12,4))

        for sensor in sensors:
            plt.plot(subset.index, subset[sensor], label=sensor)

        plt.axvline(idx, linestyle="--", linewidth=2)
        plt.title(f"Sensor behavior before failure at {idx}")
        plt.xlabel("Time")
        plt.ylabel("Sensor value")
        plt.legend()
        plt.tight_layout()
        plt.show()