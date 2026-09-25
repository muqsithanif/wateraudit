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
    # High exceedance probability and actual measured breach (BOD = 45 > 25 mg/L)
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


def test_limits_come_from_the_config_file():
    from pathlib import Path
    config = Path(__file__).resolve().parent.parent / "configs" / "limits.yaml"
    guard = RegulatoryComplianceGuard.from_yaml(config)
    assert guard.limits == {"DBO-S": 25.0, "DQO-S": 125.0, "SS-S": 35.0}
    assert (guard.ph_low, guard.ph_high) == (6.5, 8.5)
    assert (guard.watch, guard.act, guard.red) == (0.20, 0.50, 0.80)


def test_forecast_without_lab_results_is_not_flagged_by_the_lab():
    # Scoring a forecast with the lab result included would flag every
    # measured breach by definition. Without it, the band reflects only
    # what the soft sensor predicted.
    guard = RegulatoryComplianceGuard()
    probs = {"DBO-S": 0.05, "DQO-S": 0.05}
    medians = {"DBO-S": 12.0, "DQO-S": 60.0}
    measured = {"DBO-S": 40.0, "DQO-S": 60.0}

    forecast = guard.evaluate_sample(probs, medians, ph_output=7.5)
    reported = guard.evaluate_sample(probs, medians, ph_output=7.5, actual_measurements=measured)
    assert forecast.alert_band != "RED"
    assert reported.alert_band == "RED"
