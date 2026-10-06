# AI-Driven Anomaly Detection and Monitoring System

## M.Tech CSE Assignment

This project implements anomaly detection on synthetic wearable sensor data using machine learning.

## Dataset

The dataset contains 2 hours of wearable sensor readings sampled once per second.

- Total readings: 7,200
- Sampling rate: 1 reading/second
- Seed (S): 9
- Resting heart rate: 69 bpm
- Anomaly events: 20

### Anomaly Types

- Spike
- Drop-out
- Silent drift

Normal walking periods are also included in the dataset and are not considered anomalies.

## Project Structure

```text
AI-Anomaly-Detection-Assignment/
│
├── setup/
│   ├── generate_data.py
│   ├── readings.csv
│   └── signal_plot.png
│
└── question_A/
    └── level1_isolation_forest.py
