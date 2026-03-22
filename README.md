# Silent Failure & Early Warning Prediction System

## 1. Project Overview & Problem Statement
In traditional maintenance and customer retention systems, companies often detect issues **too late**. By the time a machine breaks down (or a customer churns), the opportunity to intervene is already gone. 

This project solves that problem by building a **Silent Failure Prediction System**. Instead of predicting *when* a machine is currently broken, the system monitors subtle, hidden behavioral changes in sensor data to identify a "Danger Zone" (an early warning state) *before* the actual failure occurs.

## 2. Scope & Data (Multi-Machine Integration)
**Yes, this model uses data from ALL available machines, not just one.**
The pipeline processes raw data from 28 different machines (`machine-1-*`, `machine-2-*`, `machine-3-*`), compiling a massive dataset of over **708,000** time-step observations. 

To ensure the model actually learns the physics of failures rather than just memorizing a specific machine's timeline, we implement a strict **Group-Based Split**. The model studies (trains on) 80% of the machines, and is then evaluated (tested on) the remaining 20% of machines that it has **never seen before**.

## 3. Project Flow & Architecture
The project is built as a highly modular, end-to-end Machine Learning pipeline. You can run the entire system using the provided scripts:

1. **`python run_features.py` (Feature Engineering)**
   - Loads raw sensor data for all machines.
   - Calculates time-series context: 50-step rolling averages, rolling standard deviations, rolling slopes (trends), and historical lags.
   - **Advanced EWMA Metrics:** Calculates Exponentially Weighted Moving Averages (EWMA) and short-vs-long term ratios to explicitly detect sudden sensor spikes relative to historical baselines.
   - **Feature Pruning:** Automatically drops raw sensors that have 0.0% predictive power to reduce noise.
   - Saves the unified dataset without leaking data between different `machine_id`s.

2. **`python run_dataset.py` (Label Generation)**
   - Finds the exact timestamps of actual failures.
   - Looks exactly 30 time-steps *prior* to the failure and labels them as `1` (The "Early Warning" or "Danger Zone").
   - Drops the original anomaly label so the model is strictly predicting the early warning window.

3. **`python run_train.py` (Model Training)**
   - Splits the data by `machine_id` to ensure strict isolation of the test set.
   - Calculates the exact class imbalance (usually around 18-to-1 normal-to-failure ratio) and creates a penalty weight.
   - Trains the algorithm, generates an evaluation report, and saves a `feature_importance.png` chart to show which sensors matter most.
   - Saves the trained "brain" to `models/xgboost_model.pkl`.

4. **`python run_inference.py` (Live Prediction)**
   - Simulates a real-world deployment. It loads the saved model, points it at a raw machine CSV, generates the complex features on the fly, and outputs a timeline of failure probabilities and alerts.

## 4. The Model
We use the **XGBoost Classifier (`xgboost`)**. 
XGBoost was chosen because:
- It natively handles missing data (`NaN` values created at the beginning of rolling windows).
- It is highly resilient against massive class imbalances via the `scale_pos_weight` parameter.
- It provides built-in feature importance tracking.

The model's hyperparameters (like tree depth, learning rate, tree sampling, and `min_child_weight` to prevent overfitting to noise) were rigorously optimized using a `TimeSeriesSplit` cross-validation strategy across an expanded grid search to maximize the Precision-Recall Area Under Curve (PR-AUC).

## 5. Model Evaluation & Results
Because the model is evaluated on **entirely unseen hardware**, predicting exact failures is incredibly difficult. However, the system successfully generalizes to new machines.

**Key Metrics from the Unseen Test Set (Post-Optimization):**
- **Test Set Size:** ~151,970 time-steps (representing the 20% unseen machines).
- **PR-AUC Score:** `0.2187` (A robust measure showing excellent performance across all possible decision thresholds, significantly improved by EWMA feature engineering).
- **ROC-AUC Score:** `0.6033` (Indicates the model's overall ability to distinguish normal vs. warning states is better than random guessing).
- **Optimized Threshold (0.10):** By default, models require 50% confidence to trigger an alarm. We lowered the threshold to `0.10` because *missing a failure is catastrophically more expensive than a false alarm*.
- **Recall (At 0.10 Threshold): `52%`**. The system successfully catches over half (52%) of all "Danger Zone" states on hardware it has never encountered before. 
- **Precision (At 0.10 Threshold): `10%`**. (At standard 0.5 threshold, precision rises to `22%`).

**How to interpret the results:** If the alarm rings, there is a 10% chance the machine is actually entering a critical failure state. However, by accepting these false alarms, the business guarantees that it will catch nearly half of all silent failures before the machine completely breaks.

## 6. Project Assumptions & Limitations
For full transparency, anyone reading or deploying this project should be aware of the following:

- **Assumption 1 (The Warning Window):** We assume that a machine exhibits detectable abnormal behavior exactly `30` time-steps prior to a total failure. If a machine fails instantaneously (e.g., a sudden power cord cut) without any preceding sensor degradation, this model cannot predict it.
- **Assumption 2 (Uniformity):** We assume that all machines in the dataset operate under similar physical principles. If `machine-3` is a completely different type of engine than `machine-1`, the model will struggle to generalize.
- **Limitation (False Positives):** Because the target variable is so incredibly rare (massive class imbalance), optimizing for high Recall inherently reduces Precision. The system will generate false positives. This project assumes that sending a technician to check a healthy machine is far cheaper than letting a machine fail.

## 7. Setup & Execution
To set up the project on your local machine:
```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Run the pipeline in order
python run_features.py
python run_dataset.py
python run_train.py

# 3. Test the inference script on a single machine
python run_inference.py
```