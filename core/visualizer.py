"""Dashboard for the effluent soft sensors and the forecast risk index."""
from typing import List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.softsense import ConformalInterval


class WaterAuditVisualizer:
    """Renders the three-panel test-period dashboard."""

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
        bod_limit: float = 25.0,
        cod_limit: float = 125.0,
        bands: Tuple[float, float, float] = (0.20, 0.50, 0.80),
    ) -> None:
        """Plot BOD and COD predictions against lab values, then the risk index."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=120)

        panels = [
            (axes[0], actual_bod, predicted_bod, bod_limit, "BOD", "#1f77b4", "#d90429", "o"),
            (axes[1], actual_cod, predicted_cod, cod_limit, "COD", "#2ca02c", "#e65100", "s"),
        ]
        for ax, actual, predicted, limit, name, lab_color, pred_color, marker in panels:
            ax.plot(time_index, actual, marker, linestyle="none", color=lab_color, markersize=4, alpha=0.75,
                    label=f"Lab {name}")
            ax.plot(time_index, [p.median for p in predicted], color=pred_color, lw=1.8, label="Predicted median")
            ax.fill_between(time_index, [p.lower for p in predicted], [p.upper for p in predicted],
                            color=pred_color, alpha=0.20, label="90% conformal interval")
            ax.axhline(limit, color="red", linestyle="--", lw=1.5, label=f"Limit ({limit:g} mg/L)")
            ax.set_title(f"Effluent {name}: soft-sensor prediction vs lab result", fontsize=10, weight="bold")
            ax.set_ylabel(f"{name} (mg/L)", fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3)

        watch, act, red = bands
        ax_ecri = axes[2]
        ax_ecri.plot(time_index, ecri_history, color="#3f51b5", lw=2.0, label="Risk index (forecast only)")
        ax_ecri.axhline(watch, color="#2a9d8f", linestyle=":", lw=1.2, label=f"Watch ({watch:.2f})")
        ax_ecri.axhline(act, color="#f77f00", linestyle="--", lw=1.2, label=f"Act ({act:.2f})")
        ax_ecri.axhline(red, color="#d90429", linestyle="-.", lw=1.5, label=f"Red ({red:.2f})")
        ax_ecri.axhspan(0.0, watch, facecolor="#2a9d8f", alpha=0.08)
        ax_ecri.axhspan(watch, act, facecolor="#f77f00", alpha=0.08)
        ax_ecri.axhspan(act, red, facecolor="#e76f51", alpha=0.08)
        ax_ecri.axhspan(red, 1.0, facecolor="#d90429", alpha=0.12)

        ax_ecri.set_title("Compliance risk index from the forecast", fontsize=10, weight="bold")
        ax_ecri.set_xlabel("Test-period record (time order)", fontsize=9)
        ax_ecri.set_ylabel("Risk index [0, 1]", fontsize=9)
        ax_ecri.set_ylim([-0.02, 1.02])
        ax_ecri.legend(loc="upper right", fontsize=8)
        ax_ecri.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
