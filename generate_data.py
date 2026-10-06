# ============================================================
# AI-DRIVEN ANOMALY DETECTION AND MONITORING SYSTEM
# Common Dataset Generator
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# ============================================================
# 1. PERSONAL SEED
# ============================================================

# Last four digits of USN = CS09
# Numeric seed used = 09 -> 9
S = 9

np.random.seed(S)


# ============================================================
# 2. DATASET SETTINGS
# ============================================================

N = 7200                    # 2 hours × 60 × 60
START_TIME = datetime(2026, 10, 6, 12, 0, 0)

timestamps = [
    START_TIME + timedelta(seconds=i)
    for i in range(N)
]

# Assignment requirement:
# Resting HR = 60 + (S mod 20)
RESTING_HR = 60 + (S % 20)


# ============================================================
# 3. BASE HEART RATE
# ============================================================

t = np.arange(N)

# Slow natural drift
slow_drift = 2.0 * np.sin(
    2 * np.pi * t / N
)

# Small breathing-like wave
breathing_wave = 1.5 * np.sin(
    2 * np.pi * t / 12
)

# Random sensor noise
noise = np.random.normal(
    loc=0,
    scale=1.5,
    size=N
)

# Base heart rate
hr = (
    RESTING_HR
    + slow_drift
    + breathing_wave
    + noise
)


# ============================================================
# 4. BASE ACCELEROMETER SIGNAL
# ============================================================

# Mostly low movement during normal/resting periods
acc = np.abs(
    np.random.normal(
        loc=0.08,
        scale=0.025,
        size=N
    )
)


# ============================================================
# 5. NORMAL WALKING PERIODS
#
# Walking is NOT an anomaly.
# ============================================================

walking_periods = [
    (900, 1200),      # 5 minutes
    (2700, 3300),     # 10 minutes
    (4800, 5280)      # 8 minutes
]

for start, end in walking_periods:

    duration = end - start

    # Higher accelerometer magnitude during walking
    acc[start:end] = np.abs(
        np.random.normal(
            loc=0.75,
            scale=0.15,
            size=duration
        )
    )

    # Heart rate rises by approximately 20–30 bpm
    walk_increase = np.linspace(
        20,
        25,
        duration
    )

    hr[start:end] += walk_increase

    # Normal variation while walking
    hr[start:end] += np.random.normal(
        loc=0,
        scale=2.0,
        size=duration
    )


# ============================================================
# 6. ANOMALY DEFINITIONS
#
# Total = 20 anomaly EVENTS
#
# 7 spikes
# 7 drop-outs
# 6 silent drifts
#
# All events are outside walking periods and do not overlap.
# ============================================================

anomaly_events = [

    # --------------------------------------------------------
    # SPIKES
    # Duration: 3–10 seconds
    # HR increases by >40 bpm
    # No movement
    # --------------------------------------------------------

    (500, 5, "spike"),
    (1600, 7, "spike"),
    (2400, 4, "spike"),
    (3700, 8, "spike"),
    (4400, 6, "spike"),
    (5800, 9, "spike"),
    (6500, 5, "spike"),


    # --------------------------------------------------------
    # DROP-OUTS
    # Duration: 10–30 seconds
    # Sensor reads flat or zero
    # --------------------------------------------------------

    (750, 15, "dropout"),
    (1800, 20, "dropout"),
    (2600, 12, "dropout"),
    (3850, 25, "dropout"),
    (4600, 18, "dropout"),
    (6100, 30, "dropout"),
    (6605, 22, "dropout"),


    # --------------------------------------------------------
    # SILENT DRIFTS
    # Exactly 5 minutes = 300 seconds
    # HR increases by 15 bpm
    # No movement
    # --------------------------------------------------------

    (1200, 300, "silent_drift"),
    (2000, 300, "silent_drift"),
    (3350, 300, "silent_drift"),
    (4000, 300, "silent_drift"),
    (5350, 300, "silent_drift"),
    (6900, 300, "silent_drift")
]


# ============================================================
# 7. LABEL ARRAY
# ============================================================

label = np.zeros(
    N,
    dtype=int
)

anomaly_type = np.array(
    ["normal"] * N,
    dtype=object
)


# ============================================================
# 8. INSERT ANOMALIES
# ============================================================

for start, duration, kind in anomaly_events:

    end = start + duration

    # Safety check
    if end > N:
        raise ValueError(
            f"Anomaly exceeds dataset length: {start} - {end}"
        )

    # Label the anomaly
    label[start:end] = 1
    anomaly_type[start:end] = kind


    # ========================================================
    # SPIKE
    # ========================================================

    if kind == "spike":

        # >40 bpm increase
        spike_height = np.random.uniform(
            45,
            60
        )

        hr[start:end] += spike_height

        # No movement
        acc[start:end] = np.abs(
            np.random.normal(
                0.05,
                0.01,
                end - start
            )
        )


    # ========================================================
    # DROP-OUT
    # ========================================================

    elif kind == "dropout":

        # Sensor may read zero or a flat value
        dropout_value = np.random.choice(
            [0.0, RESTING_HR]
        )

        hr[start:end] = dropout_value

        # Very little movement
        acc[start:end] = np.abs(
            np.random.normal(
                0.05,
                0.01,
                end - start
            )
        )


    # ========================================================
    # SILENT DRIFT
    # ========================================================

    elif kind == "silent_drift":

        drift_length = end - start

        # Exactly +15 bpm over 5 minutes
        drift = np.linspace(
            0,
            15,
            drift_length
        )

        # Use local HR as starting point
        base_hr = hr[start]

        hr[start:end] = (
            base_hr
            + drift
            + np.random.normal(
                0,
                0.4,
                drift_length
            )
        )

        # No movement
        acc[start:end] = np.abs(
            np.random.normal(
                0.05,
                0.01,
                drift_length
            )
        )


# ============================================================
# 9. CREATE DATAFRAME
# ============================================================

readings = pd.DataFrame({
    "timestamp": timestamps,
    "hr": hr,
    "acc": acc,
    "label": label,
    "anomaly_type": anomaly_type
})


# ============================================================
# 10. VERIFY DATASET
# ============================================================

print("=" * 65)
print("WEARABLE SENSOR DATA GENERATED")
print("=" * 65)

print(f"Seed (S): {S}")
print(f"Resting HR: {RESTING_HR} bpm")
print(f"Number of rows: {len(readings)}")
print(f"Number of anomaly events: {len(anomaly_events)}")
print(f"Number of anomalous readings: {label.sum()}")

print("\nAnomaly EVENT counts:")

event_counts = {}

for _, _, kind in anomaly_events:
    event_counts[kind] = event_counts.get(kind, 0) + 1

for kind, count in event_counts.items():
    print(f"{kind:15s}: {count}")

print("\nAnomalous READING counts:")

print(
    readings[
        readings["label"] == 1
    ]["anomaly_type"].value_counts()
)

print("\nWalking periods:")

for start, end in walking_periods:
    print(
        f"{start}s - {end}s "
        f"({(end - start) / 60:.1f} minutes)"
    )


# ============================================================
# 11. SAVE CSV
# ============================================================

readings.to_csv(
    "readings.csv",
    index=False
)

print("\nCSV saved as: readings.csv")


# ============================================================
# 12. CREATE FULL SIGNAL PLOT
# ============================================================

plt.figure(
    figsize=(16, 7)
)

# Heart-rate signal
plt.plot(
    readings["timestamp"],
    readings["hr"],
    linewidth=0.8,
    label="Heart Rate"
)


# Mark anomaly readings
anomaly_mask = readings["label"] == 1

plt.scatter(
    readings.loc[anomaly_mask, "timestamp"],
    readings.loc[anomaly_mask, "hr"],
    s=8,
    label="Anomaly"
)


# Mark walking periods
for i, (start, end) in enumerate(walking_periods):

    plt.axvspan(
        readings.loc[start, "timestamp"],
        readings.loc[end - 1, "timestamp"],
        alpha=0.15,
        label="Walking" if i == 0 else None
    )


plt.title(
    f"Wearable Heart Rate Signal "
    f"(Seed S={S})"
)

plt.xlabel("Time")
plt.ylabel("Heart Rate (bpm)")

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 13. SAVE PLOT
# ============================================================

plt.savefig(
    "signal_plot.png",
    dpi=200
)

plt.show()

print("Plot saved as: signal_plot.png")

print("=" * 65)
print("DATA GENERATION COMPLETE")
print("=" * 65)
