"""Execution script: End-to-end soft-sensing, compliance monitoring, and diagnostic dashboard."""
import time
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from core.schema import WWTPDatasetLoader
from core.softsense import QuantileSoftSensor
from core.compliance import RegulatoryComplianceGuard
from core.efficiency import SettlerEfficiencyAnalyzer
from core.visualizer import WaterAuditVisualizer


def main():
    print("==================================================================")
    print("    wateraudit: Wastewater Effluent Compliance & Soft-Sensing")
    print("==================================================================")

    # 1. Load Authentic UCI Benchmark Dataset
    print("\n[1/5] Loading Authentic UCI Wastewater Treatment Plant Dataset...")
    df_raw = WWTPDatasetLoader.load_raw()
    print(f"      Loaded Total Records      : {len(df_raw)} operational days across {len(df_raw.columns) - 1} sensors")

    X, Y = WWTPDatasetLoader.get_feature_matrix(df_raw, include_tier_1=True)
    print(f"      Constructed Feature Matrix: {X.shape[1]} leak-free features (Tier 0 & Tier 1)")

    # 2. Chronological Train / Calibration / Test Split
    n_total = len(df_raw)
    n_train = int(n_total * 0.60)
    n_cal = int(n_total * 0.20)
    n_test = n_total - n_train - n_cal

    X_train, Y_train = X.iloc[:n_train], Y.iloc[:n_train]
    X_cal, Y_cal = X.iloc[n_train:n_train + n_cal], Y.iloc[n_train:n_train + n_cal]
    X_test, Y_test = X.iloc[n_train + n_cal:], Y.iloc[n_train + n_cal:]

    print(f"      Chronological Split       : Train={len(X_train)}, Cal={len(X_cal)}, Test={len(X_test)} days")

    # 3. Fit Quantile Soft-Sensors & Conformal Calibration
    print("\n[2/5] Training Quantile Boosted Trees & Conformalizing 90% Intervals...")
    sensor_bod = QuantileSoftSensor(target_name="DBO-S", alpha=0.10, random_state=42)
    sensor_cod = QuantileSoftSensor(target_name="DQO-S", alpha=0.10, random_state=42)

    t0_fit = time.perf_counter()
    sensor_bod.fit(X_train, Y_train["DBO-S"])
    sensor_bod.calibrate(X_cal, Y_cal["DBO-S"])

    sensor_cod.fit(X_train, Y_train["DQO-S"])
    sensor_cod.calibrate(X_cal, Y_cal["DQO-S"])
    fit_time = time.perf_counter() - t0_fit

    print(f"      Models Trained in         : {fit_time:.2f} s")
    print(f"      BOD Conformal Order Stat  : q_hat = {sensor_bod.conformal_q_hat:.4f}")
    print(f"      COD Conformal Order Stat  : q_hat = {sensor_cod.conformal_q_hat:.4f}")

    # 4. Out-of-Sample Inference on Test Set
    print("\n[3/5] Inferring Test Period & Calculating Conformal Intervals...")
    bod_intervals = sensor_bod.predict_interval(X_test, regulatory_limit=30.0)
    cod_intervals = sensor_cod.predict_interval(X_test, regulatory_limit=125.0)

    # Evaluate Empirical Coverage on valid test points
    y_test_bod = Y_test["DBO-S"].values
    y_test_cod = Y_test["DQO-S"].values

    valid_bod = np.isfinite(y_test_bod)
    covered_bod = [
        (bod_intervals[i].lower <= y_test_bod[i] <= bod_intervals[i].upper)
        for i in range(len(y_test_bod)) if valid_bod[i]
    ]
    cov_bod_pct = (sum(covered_bod) / len(covered_bod)) * 100.0

    valid_cod = np.isfinite(y_test_cod)
    covered_cod = [
        (cod_intervals[i].lower <= y_test_cod[i] <= cod_intervals[i].upper)
        for i in range(len(y_test_cod)) if valid_cod[i]
    ]
    cov_cod_pct = (sum(covered_cod) / len(covered_cod)) * 100.0

    print(f"      BOD Empirical Test Coverage: {cov_bod_pct:.1f}% (Nominal Guarantee: 90.0%)")
    print(f"      COD Empirical Test Coverage: {cov_cod_pct:.1f}% (Nominal Guarantee: 90.0%)")

    # 5. Regulatory Compliance Guard & Settler Efficiency Diagnostics
    print("\n[4/5] Evaluating Discharge Compliance & Settler Health...")
    compliance_guard = RegulatoryComplianceGuard()
    efficiency_analyzer = SettlerEfficiencyAnalyzer()

    ecri_history = []
    alert_bands = []
    active_alarms_total = 0

    df_test_full = df_raw.iloc[n_train + n_cal:]

    for i in range(len(X_test)):
        row = df_test_full.iloc[i]

        pred_exceed = {
            "DBO-S": bod_intervals[i].exceedance_prob,
            "DQO-S": cod_intervals[i].exceedance_prob,
        }
        pred_med = {
            "DBO-S": bod_intervals[i].median,
            "DQO-S": cod_intervals[i].median,
        }

        actual_meas = {
            "DBO-S": row["DBO-S"],
            "DQO-S": row["DQO-S"],
            "SS-S": row["SS-S"],
        }

        status = compliance_guard.evaluate_sample(
            predicted_exceed_probs=pred_exceed,
            predicted_medians=pred_med,
            ph_output=row["PH-S"] if np.isfinite(row["PH-S"]) else 7.5,
            actual_measurements=actual_meas,
        )
        ecri_history.append(status.ecri_score)
        alert_bands.append(status.alert_band)

        diag = efficiency_analyzer.evaluate_stages(
            q_in=row["Q-E"],
            ss_in=row["SS-E"],
            ss_primary=row["SS-P"],
            ss_out=row["SS-S"],
            bod_in=row["DBO-E"],
            bod_out=row["DBO-S"],
            cod_in=row["DQO-E"],
            cod_out=row["DQO-S"],
            sed_out=row["SED-S"],
        )
        if len(diag.active_alarms) > 0:
            active_alarms_total += 1

    print(f"      Average Test ECRI Score   : {np.mean(ecri_history):.3f}")
    print(f"      Green / Watch / Act / Red : {alert_bands.count('GREEN')} / {alert_bands.count('WATCH')} / {alert_bands.count('ACT')} / {alert_bands.count('RED')} days")
    print(f"      Total Settler Alarms Raised: {active_alarms_total} events")

    # 6. Export Visual Dashboard
    print("\n[5/5] Exporting Environmental Operational Dashboard...")
    samples_dir = PROJECT_ROOT / "samples"
    samples_dir.mkdir(exist_ok=True)
    dashboard_path = samples_dir / "water_audit_dashboard.png"

    time_idx = np.arange(len(X_test))
    WaterAuditVisualizer.plot_effluent_dashboard(
        time_index=time_idx,
        actual_bod=y_test_bod,
        predicted_bod=bod_intervals,
        actual_cod=y_test_cod,
        predicted_cod=cod_intervals,
        ecri_history=ecri_history,
        alert_bands=alert_bands,
        save_path=str(dashboard_path),
    )
    print(f"      Saved Dashboard Plot      -> {dashboard_path}")

    print("\n==================================================================")
    print("Water Audit Pipeline Completed Successfully!")
    print("==================================================================")


if __name__ == "__main__":
    main()
