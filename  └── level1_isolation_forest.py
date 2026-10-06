import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# 1. SETTINGS
# ============================================================

S = 9

data = pd.read_csv("../../setup/readings.csv")


# ============================================================
# 2. FEATURES
# ============================================================

# Isolation Forest does NOT use:
# label
# anomaly_type
#
# These are used only later for evaluation.

X = data[["hr", "acc"]].values


# ============================================================
# 3. ISOLATION FOREST
# ============================================================

model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=S
)

model.fit(X)


# ============================================================
# 4. PREDICTIONS
# ============================================================

# Isolation Forest:
# -1 = anomaly
#  1 = normal

raw_prediction = model.predict(X)

y_pred = (raw_prediction == -1).astype(int)


# ============================================================
# 5. OVERALL GROUND TRUTH
# ============================================================

y_true = data["label"].astype(int).values


# ============================================================
# 6. OVERALL METRICS
# ============================================================

overall_precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

overall_recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

overall_f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


print("=" * 65)
print("QUESTION A - LEVEL 1")
print("ISOLATION FOREST")
print("=" * 65)

print(f"Seed: {S}")
print(f"Total readings: {len(data)}")
print(f"Actual anomalous readings: {y_true.sum()}")
print(f"Predicted anomalous readings: {y_pred.sum()}")

print("\nOverall performance:")
print(f"Precision: {overall_precision:.4f}")
print(f"Recall:    {overall_recall:.4f}")
print(f"F1 score:  {overall_f1:.4f}")


# ============================================================
# 7. PER-ANOMALY-TYPE METRICS
# ============================================================

print("\n" + "=" * 65)
print("PER-ANOMALY-TYPE RESULTS")
print("=" * 65)

anomaly_types = [
    "spike",
    "dropout",
    "silent_drift"
]

for anomaly_type in anomaly_types:

    # 1 for this anomaly type, 0 for everything else
    y_type_true = (
        data["anomaly_type"] == anomaly_type
    ).astype(int).values

    # SAME Isolation Forest prediction is used.
    # We do NOT restrict it using the actual anomaly type.
    y_type_pred = y_pred

    p = precision_score(
        y_type_true,
        y_type_pred,
        zero_division=0
    )

    r = recall_score(
        y_type_true,
        y_type_pred,
        zero_division=0
    )

    f = f1_score(
        y_type_true,
        y_type_pred,
        zero_division=0
    )

    actual_count = y_type_true.sum()
    detected_count = np.sum(
        (y_type_true == 1) &
        (y_type_pred == 1)
    )

    print(f"\n{anomaly_type.upper()}")
    print(f"Actual readings:    {actual_count}")
    print(f"Detected readings:  {detected_count}")
    print(f"Precision:          {p:.4f}")
    print(f"Recall:             {r:.4f}")
    print(f"F1 score:           {f:.4f}")


# ============================================================
# 8. WALKING FALSE ALARMS
# ============================================================

walking_mask = np.zeros(
    len(data),
    dtype=bool
)

walking_periods = [
    (900, 1200),
    (2700, 3300),
    (4800, 5280)
]

for start, end in walking_periods:
    walking_mask[start:end] = True


walking_false_alarms = np.sum(
    (y_pred == 1) &
    (y_true == 0) &
    walking_mask
)


print("\n" + "=" * 65)
print("FALSE ALARMS DURING WALKING")
print("=" * 65)

print(
    f"False alarms during walking: "
    f"{walking_false_alarms}"
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results = data.copy()

results["isolation_forest_prediction"] = y_pred
results["isolation_forest_score"] = model.decision_function(X)

results.to_csv(
    "level1_predictions.csv",
    index=False
)

print("\nPredictions saved as:")
print("level1_predictions.csv")

print("=" * 65)