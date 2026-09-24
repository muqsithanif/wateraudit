"""Visualizer for wastewater effluent compliance, soft-sensor conformal intervals, and plant telemetry."""
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.softsense import ConformalInterval


class WaterAuditVisualizer:
    """Renders comprehensive environmental compliance dashboards and soft-sensing validation plots."""

    @classmethod
    def plot_effluent_dashboard(
        cls,
        time_index: np.ndarray,
        actual_bod: np.ndarray,
        predicted_bod: List[ConformalInterval],
        actual_cod: np.ndarray,
        predicted_cod: List[ConformalInterval],
        ecri_history: List[float],
        alert_bands: List[str],
        save_path: str,
    ) -> None:
        """Plot a 3-panel operational dashboard: BOD prediction, COD prediction, and ECRI risk timeline."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=120)

        # 1. Biological Oxygen Demand (BOD) Panel
        ax_bod = axes[0]
        bod_meds = [p.median for p in predicted_bod]
        bod_lows = [p.lower for p in predicted_bod]
        bod_highs = [p.upper for p in predicted_bod]

        ax_bod.plot(time_index, actual_bod, "o", color="#1f77b4", markersize=4, alpha=0.75, label="Lab BOD Measurement")
        ax_bod.plot(time_index, bod_meds, color="#d90429", lw=1.8, label="Soft-Sensor Predicted Median")
        ax_bod.fill_between(time_index, bod_lows, bod_highs, color="#d90429", alpha=0.20, label="90% Conformal Bounds")
        ax_bod.axhline(30.0, color="red", linestyle="--", lw=1.5, label="Legal Limit (30 mg/L)")

        ax_bod.set_title("Effluent Biological Oxygen Demand (BOD / DBO-S) Soft-Sensing", fontsize=10, weight="bold")
        ax_bod.set_ylabel("BOD (mg/L)", fontsize=9)
        ax_bod.legend(loc="upper right", fontsize=8)
        ax_bod.grid(alpha=0.3)

        # 2. Chemical Oxygen Demand (COD) Panel
        ax_cod = axes[1]
        cod_meds = [p.median for p in predicted_cod]
        cod_lows = [p.lower for p in predicted_cod]
        cod_highs = [p.upper for p in predicted_cod]

        ax_cod.plot(time_index, actual_cod, "s", color="#2ca02c", markersize=4, alpha=0.75, label="Lab COD Measurement")
        ax_cod.plot(time_index, cod_meds, color="#e65100", lw=1.8, label="Soft-Sensor Predicted Median")
        ax_cod.fill_between(time_index, cod_lows, cod_highs, color="#e65100", alpha=0.20, label="90% Conformal Bounds")
        ax_cod.axhline(125.0, color="red", linestyle="--", lw=1.5, label="Legal Limit (125 mg/L)")

        ax_cod.set_title("Effluent Chemical Oxygen Demand (COD / DQO-S) Soft-Sensing", fontsize=10, weight="bold")
        ax_cod.set_ylabel("COD (mg/L)", fontsize=9)
        ax_cod.legend(loc="upper right", fontsize=8)
        ax_cod.grid(alpha=0.3)

        # 3. Environmental Compliance Risk Index (ECRI)
        ax_ecri = axes[2]
        ax_ecri.plot(time_index, ecri_history, color="#3f51b5", lw=2.0, label="Daily ECRI Score")
        ax_ecri.axhline(0.20, color="#2a9d8f", linestyle=":", lw=1.2, label="Green/Watch (0.20)")
        ax_ecri.axhline(0.50, color="#f77f00", linestyle="--", lw=1.2, label="Watch/Act (0.50)")
        ax_ecri.axhline(0.80, color="#d90429", linestyle="-.", lw=1.5, label="Critical Red (0.80)")

        # Color-coded background bands
        ax_ecri.axhspan(0.0, 0.20, facecolor="#2a9d8f", alpha=0.08)
        ax_ecri.axhspan(0.20, 0.50, facecolor="#f77f00", alpha=0.08)
        ax_ecri.axhspan(0.50, 0.80, facecolor="#e76f51", alpha=0.08)
        ax_ecri.axhspan(0.80, 1.0, facecolor="#d90429", alpha=0.12)

        ax_ecri.set_title("Environmental Compliance Risk Index (ECRI) & Regulatory Alert Level", fontsize=10, weight="bold")
        ax_ecri.set_xlabel("Operational Days (Timeline)", fontsize=9)
        ax_ecri.set_ylabel("ECRI Score [0, 1]", fontsize=9)
        ax_ecri.set_ylim([-0.02, 1.02])
        ax_ecri.legend(loc="upper right", fontsize=8)
        ax_ecri.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
