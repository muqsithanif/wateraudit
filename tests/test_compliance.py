"""Unit tests for regulatory compliance checking and ECRI calculation."""
import pytest
from core.compliance import RegulatoryComplianceGuard


def test_compliant_sample():
    guard = RegulatoryComplianceGuard()
    probs = {"DBO-S": 0.05, "DQO-S": 0.04, "SS-S": 0.02}
    medians = {"DBO-S": 15.0, "DQO-S": 60.0, "SS-S": 18.0}

    status = guard.evaluate_sample(
        predicted_exceed_probs=probs,
        predicted_medians=medians,
        ph_output=7.6,
    )
    assert status.alert_band in ["GREEN", "WATCH"]
    assert status.ecri_score < 0.30
    assert len(status.violations) == 0


def test_severe_violation_triggers_red():
    guard = RegulatoryComplianceGuard()
    # High exceedance probability and actual measured breach (BOD = 45 > 30 mg/L)
    probs = {"DBO-S": 0.95, "DQO-S": 0.88, "SS-S": 0.80}
    medians = {"DBO-S": 48.0, "DQO-S": 160.0, "SS-S": 55.0}
    actual = {"DBO-S": 45.0, "DQO-S": 150.0, "SS-S": 52.0}

    status = guard.evaluate_sample(
        predicted_exceed_probs=probs,
        predicted_medians=medians,
        ph_output=7.4,
        actual_measurements=actual,
    )
    assert status.alert_band == "RED"
    assert status.ecri_score >= 0.80
    assert len(status.violations) > 0


def test_ph_out_of_bounds_triggers_violation():
    guard = RegulatoryComplianceGuard()
    probs = {"DBO-S": 0.05, "DQO-S": 0.05}
    medians = {"DBO-S": 15.0, "DQO-S": 60.0}

    # Acidic breach (pH 5.8 < 6.5)
    status = guard.evaluate_sample(
        predicted_exceed_probs=probs,
        predicted_medians=medians,
        ph_output=5.8,
    )
    assert status.alert_band == "RED"
    assert any("pH" in v for v in status.violations)
