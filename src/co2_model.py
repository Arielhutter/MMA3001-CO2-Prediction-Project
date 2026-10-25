"""Reusable functions for the MMA3001 CO2 soft-sensing project."""
import json

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler


TARGET = "co2"
FEATURES = ["humidity", "pressure", "temperature", "voc",
            "pm1", "pm4", "pm2_5", "pm10", "hour"]
TRAIN_FRAC = 0.8


def parse_jsondata(row):
    """Unpack one raw row into a flat dict.

    Parameters
    ----------
    row : mapping
        Needs 'sensorid', 'createdate' and a JSON string in 'jsondata'.

    Returns
    -------
    dict
        'sensorid', 'timestamp', 'device_time' and one key per variable.
        Variables with no values are skipped.
    """
    readings = json.loads(row["jsondata"])
    parsed = {"sensorid": row["sensorid"], "timestamp": row["createdate"],
              "device_time": None}
    for item in readings:
        values = item.get("values", [])
        if not values:
            continue
        parsed[item["variable"]["name"]] = values[0].get("value")
        parsed["device_time"] = values[0].get("time")
    return parsed


def deduplicate(df):
    """Drop re-inserted readings, keeping the first row per (sensorid, device_time)."""
    return df.drop_duplicates(subset=["sensorid", "device_time"],
                              keep="first").reset_index(drop=True)


def chronological_split(df, sensor_id, train_frac=TRAIN_FRAC):
    """Split one sensor's rows by time: first train_frac for training, the rest for testing.

    Returns
    -------
    tuple of pandas.DataFrame
        (train, test), sorted by 'timestamp' with no overlap.
    """
    sub = df[df["sensorid"] == sensor_id].sort_values("timestamp")
    k = int(len(sub) * train_frac)
    return sub.iloc[:k], sub.iloc[k:]


def scale_features(train, test, features=FEATURES):
    """Standardise inputs using the training data only (no leakage).

    Returns
    -------
    tuple
        (X_train, X_test, fitted StandardScaler)
    """
    scaler = StandardScaler()
    return scaler.fit_transform(train[features]), scaler.transform(test[features]), scaler


def score(y_true, y_pred):
    """Return RMSE (ppm), MAE (ppm) and R^2 for one set of predictions."""
    return {"rmse": root_mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)}