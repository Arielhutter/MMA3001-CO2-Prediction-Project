import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from src.co2_model import (chronological_split, deduplicate, parse_jsondata,
                           scale_features, score)


def make_row(values):
    payload = [{"variable": {"name": n}, "values": [{"time": 100, "value": v}] if v is not None else []}
               for n, v in values.items()]
    return {"sensorid": 1, "createdate": "2024-01-01", "jsondata": json.dumps(payload)}


def test_parse_extracts_values_and_skips_empty():
    out = parse_jsondata(make_row({"Temperature": 22.5, "Formaldehyde": None}))
    assert out["Temperature"] == 22.5
    assert "Formaldehyde" not in out
    assert out["device_time"] == 100


def test_deduplicate_keeps_first_of_each_key():
    df = pd.DataFrame({"sensorid": [1, 1, 1, 2], "device_time": [10, 10, 11, 10],
                       "co2": [400, 400, 410, 420]})
    out = deduplicate(df)
    assert len(out) == 3
    assert not out.duplicated(subset=["sensorid", "device_time"]).any()


def test_split_is_chronological_with_no_overlap():
    t = pd.date_range("2024-01-01", periods=100, freq="10min")
    df = pd.DataFrame({"sensorid": 1, "timestamp": t}).sample(frac=1, random_state=0)  # shuffled
    train, test = chronological_split(df, 1)
    assert len(train) == 80 and len(test) == 20
    assert train["timestamp"].max() < test["timestamp"].min()


def test_scaler_is_fitted_on_training_data_only():
    train = pd.DataFrame({"a": [0.0, 2.0, 4.0]})
    test = pd.DataFrame({"a": [10.0, 12.0]})
    X_tr, X_te, _ = scale_features(train, test, features=["a"])
    assert np.isclose(X_tr.mean(), 0) and np.isclose(X_tr.std(), 1)
    assert X_te.mean() > 3          # test data is NOT re-centred on itself


def test_score_matches_hand_calculation():
    # errors 0, 0, -2: RMSE = sqrt(4/3), MAE = 2/3, R2 = 1 - 4/2 = -1
    m = score(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert np.isclose(m["rmse"], np.sqrt(4 / 3))
    assert np.isclose(m["mae"], 2 / 3)
    assert np.isclose(m["r2"], -1.0)


def test_constant_mean_prediction_has_zero_r2():
    y = np.array([400.0, 450.0, 500.0])
    assert np.isclose(score(y, np.full(3, y.mean()))["r2"], 0.0)


def test_linear_regression_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 5
    lr = LinearRegression().fit(X, y)
    assert np.allclose(lr.coef_, [3, -2]) and np.isclose(lr.intercept_, 5)


def test_svr_fits_a_known_smooth_function():
    X = np.linspace(0, 1, 100).reshape(-1, 1)
    y = 3 * X.ravel()
    pred = SVR(kernel="rbf", C=100, gamma=1, epsilon=0.01).fit(X, y).predict(X)
    assert score(y, pred)["rmse"] < 0.1