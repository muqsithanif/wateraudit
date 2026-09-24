"""Unit tests for quantile soft-sensing and conformal prediction bounds."""
import numpy as np
import pandas as pd
import pytest
from core.softsense import QuantileSoftSensor


@pytest.fixture
def dummy_data():
    rng = np.random.RandomState(42)
    n = 150
    X = pd.DataFrame({
        "Q-E": rng.normal(35000, 5000, n),
        "PH-E": rng.normal(7.8, 0.3, n),
        "COND-E": rng.normal(2000, 300, n),
    })
    # Target BOD with log-normal noise
    y = pd.Series(np.clip(20.0 + 0.0003 * X["Q-E"] + rng.normal(0, 4, n), 2.0, 80.0))
    return X, y


def test_soft_sensor_fit_calibrate_predict(dummy_data):
    X, y = dummy_data
    sensor = QuantileSoftSensor(target_name="DBO-S", alpha=0.10, random_state=42)

    sensor.fit(X.iloc[:100], y.iloc[:100])
    sensor.calibrate(X.iloc[100:130], y.iloc[100:130])

    assert sensor.is_calibrated is True
    assert sensor.conformal_q_hat > 0.0

    intervals = sensor.predict_interval(X.iloc[130:], regulatory_limit=30.0)
    assert len(intervals) == 20

    for iv in intervals:
        # Quantile uncrossing invariant: lower <= median <= upper
        assert iv.lower <= iv.median <= iv.upper
        # Physical non-negativity constraint
        assert iv.lower >= 0.0
        # Exceedance probability bounded in [0, 1]
        assert 0.0 <= iv.exceedance_prob <= 1.0
