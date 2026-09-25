"""Train the effluent BOD/COD soft sensors on the UCI plant data and score the test period.

    python scripts/run_audit.py                  # chronological split (default)
    python scripts/run_audit.py --split random   # shuffled split, for comparison

Writes a JSON summary to results/, and for the chronological split also
samples/water_audit_dashboard.png.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score

from core.schema import TIER_2_LAGGED_LAB, WWTPDatasetLoader
from core.softsense import QuantileSoftSensor
from core.compliance import RegulatoryComplianceGuard
from core.efficiency import SettlerEfficiencyAnalyzer
from core.visualizer import WaterAuditVisualizer

TARGETS = {"BOD": "DBO-S", "COD": "DQO-S"}


def split_indices(n: int, mode: str, seed: int):
    """60/20/20 train/calibration/test split, in time order or shuffled."""
    n_train, n_cal = int(n * 0.60), int(n * 0.20)
    order = np.arange(n) if mode == "chronological" else np.random.default_rng(seed).permutation(n)
    train, cal, test = order[:n_train], order[n_train:n_train + n_cal], order[n_train + n_cal:]
    return np.sort(train), np.sort(cal), np.sort(test)


def coverage(intervals, actual: np.ndarray) -> float:
    hits = [iv.lower <= y <= iv.upper for iv, y in zip(intervals, actual) if np.isfinite(y)]
    return float(np.mean(hits))


def diagnose(df, rows, analyzer):
    reports = []
    for _, row in df.iloc[rows].iterrows():
        reports.append(analyzer.evaluate_stages(
            q_in=row["Q-E"], ss_in=row["SS-E"], ss_primary=row["SS-P"], ss_out=row["SS-S"],
            bod_in=row["DBO-E"], bod_out=row["DBO-S"], cod_in=row["DQO-E"], cod_out=row["DQO-S"],
            sed_out=row["SED-S"],
        ))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["chronological", "random"], default="chronological")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed for --split random")
    args = parser.parse_args()

    guard = RegulatoryComplianceGuard.from_yaml(PROJECT_ROOT / "configs" / "limits.yaml")
    df = WWTPDatasetLoader.load_raw()
    X, Y = WWTPDatasetLoader.get_feature_matrix(df, include_tier_1=True)
    train, cal, test = split_indices(len(df), args.split, args.seed)
    print(f"UCI water treatment plant: {len(df)} records, {X.shape[1]} features")
    print(f"{args.split} split: train={len(train)}, calibration={len(cal)}, test={len(test)}")

    # 1. Soft sensors with 90% conformalized quantile intervals, compared with
    # the naive alternative of using the latest lab result that is available.
    intervals, cov, mae = {}, {}, {}
    for name, col in TARGETS.items():
        sensor = QuantileSoftSensor(target_name=col, alpha=0.10, random_state=42)
        sensor.fit(X.iloc[train], Y[col].iloc[train])
        sensor.calibrate(X.iloc[cal], Y[col].iloc[cal])
        intervals[col] = sensor.predict_interval(X.iloc[test], regulatory_limit=guard.limits[col])
        actual = Y[col].iloc[test].to_numpy()
        cov[name] = coverage(intervals[col], actual)

        median = np.array([iv.median for iv in intervals[col]])
        last_known = df[col].shift(TIER_2_LAGGED_LAB[col]).iloc[test].to_numpy()
        both = np.isfinite(actual) & np.isfinite(last_known)
        mae[name] = {
            "soft_sensor_median": round(float(np.mean(np.abs(median[both] - actual[both]))), 2),
            "latest_lab_value": round(float(np.mean(np.abs(last_known[both] - actual[both]))), 2),
        }
        print(f"{name}: coverage of the 90% interval = {cov[name]:.1%}; MAE {mae[name]['soft_sensor_median']} mg/L "
              f"vs {mae[name]['latest_lab_value']} mg/L for the latest known lab value")

    # 2. Risk index from the forecast alone: soft-sensor output plus the online
    # pH probe, never the lab result it is meant to anticipate.
    test_rows = df.iloc[test]
    ecri, bands = [], []
    for i in range(len(test)):
        ph = test_rows["PH-S"].iloc[i]
        status = guard.evaluate_sample(
            predicted_exceed_probs={col: intervals[col][i].exceedance_prob for col in TARGETS.values()},
            predicted_medians={col: intervals[col][i].median for col in TARGETS.values()},
            ph_output=ph if np.isfinite(ph) else 7.5,
        )
        ecri.append(status.ecri_score)
        bands.append(status.alert_band)
    bands_arr = np.array(bands)

    # 3. Score the forecast against lab-confirmed BOD/COD breaches
    breach = np.zeros(len(test), dtype=bool)
    for col in TARGETS.values():
        measured = test_rows[col].to_numpy()
        breach |= np.isfinite(measured) & (measured > guard.limits[col])
    alarm = np.isin(bands_arr, ["ACT", "RED"])
    watch_or_above = np.isin(bands_arr, ["WATCH", "ACT", "RED"])
    band_counts = {b: int((bands_arr == b).sum()) for b in ["GREEN", "WATCH", "ACT", "RED"]}
    early_warning = {
        "breach_days": int(breach.sum()),
        "breaches_flagged_act_or_red": int((alarm & breach).sum()),
        "breaches_flagged_watch_or_above": int((watch_or_above & breach).sum()),
        "act_or_red_days": int(alarm.sum()),
        "act_or_red_without_breach": int((alarm & ~breach).sum()),
        # Threshold-free: how well the index ranks breach days above the others.
        "risk_index_auroc": (round(float(roc_auc_score(breach, ecri)), 3)
                             if 0 < breach.sum() < len(breach) else None),
    }
    print(f"forecast bands on test days: {band_counts}")
    print(f"lab-confirmed breach days: {early_warning['breach_days']}, "
          f"flagged ACT/RED: {early_warning['breaches_flagged_act_or_red']}, "
          f"flagged WATCH or above: {early_warning['breaches_flagged_watch_or_above']}")
    print(f"ACT/RED days without a breach: {early_warning['act_or_red_without_breach']} "
          f"of {early_warning['act_or_red_days']}; risk-index AUROC: {early_warning['risk_index_auroc']}")

    summary = {
        "split": args.split,
        "seed": args.seed if args.split == "random" else None,
        "records": {"train": len(train), "calibration": len(cal), "test": len(test)},
        "limits_mg_per_l": guard.limits,
        "interval_coverage_target": 0.90,
        "interval_coverage": {k: round(v, 4) for k, v in cov.items()},
        "median_abs_error_mg_per_l": mae,
        "forecast_bands": band_counts,
        "early_warning": early_warning,
    }

    if args.split == "chronological":
        # 4. Stage diagnostics. The "normal" radius is set from the training
        # period only, the same way a threshold would be set at commissioning.
        baseline = diagnose(df, train, SettlerEfficiencyAnalyzer(normal_radius=np.inf))
        radius = float(np.quantile([r.deviation for r in baseline], 0.975))
        reports = diagnose(df, test, SettlerEfficiencyAnalyzer(normal_radius=radius))
        top = [r.fault_hypotheses[0][0] for r in reports]
        summary["stage_diagnostics"] = {
            "normal_radius": round(radius, 3),
            "cusum_alarm_days": int(sum(bool(r.active_alarms) for r in reports)),
            "top_hypothesis_counts": {h: top.count(h) for h in dict.fromkeys(top)},
        }
        print(f"stage diagnostics on test days: {summary['stage_diagnostics']['top_hypothesis_counts']}, "
              f"CUSUM alarm days: {summary['stage_diagnostics']['cusum_alarm_days']}")

        dashboard = PROJECT_ROOT / "samples" / "water_audit_dashboard.png"
        dashboard.parent.mkdir(exist_ok=True)
        WaterAuditVisualizer.plot_effluent_dashboard(
            time_index=np.arange(len(test)),
            actual_bod=test_rows["DBO-S"].to_numpy(),
            predicted_bod=intervals["DBO-S"],
            actual_cod=test_rows["DQO-S"].to_numpy(),
            predicted_cod=intervals["DQO-S"],
            ecri_history=ecri,
            alert_bands=bands,
            save_path=str(dashboard),
            bod_limit=guard.limits["DBO-S"],
            cod_limit=guard.limits["DQO-S"],
            bands=(guard.watch, guard.act, guard.red),
        )
        print(f"dashboard -> {dashboard.relative_to(PROJECT_ROOT)}")

    name = "summary_chronological.json" if args.split == "chronological" else f"summary_random_seed{args.seed}.json"
    out = PROJECT_ROOT / "results" / name
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary -> {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
