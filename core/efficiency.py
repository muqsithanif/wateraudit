"""Settler stage efficiency decomposition, CUSUM monitoring, and fault signature matching."""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class PlantDiagnosticReport:
    global_bod_removal: float
    global_cod_removal: float
    global_ss_removal: float
    primary_efficiency_residual: float
    secondary_efficiency_residual: float
    active_alarms: List[str]
    deviation: float                            # Norm of the diagnostic vector
    fault_hypotheses: List[Tuple[str, float]]  # [(Hypothesis, Confidence)]


class SettlerEfficiencyAnalyzer:
    """Isolates operational unit failures (primary vs secondary settlers) using log-ratio decomposition."""

    FAULT_PROTOTYPES = {
        "Hydraulic Shock Load": np.array([1.0, 1.0, 1.0, 0.8, 0.0]),
        "Sludge Bulking": np.array([0.0, 0.0, 1.5, 0.2, 1.2]),
        "Toxic Shock / Bio-Inhibition": np.array([0.0, 1.4, 0.1, 0.0, 0.1]),
        "Primary Settler Mechanical Fault": np.array([0.0, 0.2, 0.2, -1.0, 0.0]),
    }

    def __init__(self, cusum_k: float = 0.5, cusum_h: float = 4.0, normal_radius: float = 1.0):
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self.normal_radius = normal_radius
        self.cusum_primary: float = 0.0
        self.cusum_secondary: float = 0.0

    @staticmethod
    def log_ratio_efficiency(cin: float, cout: float) -> float:
        """Compute stage log-ratio: l = ln(C_out / C_in). More negative = better removal."""
        if cin <= 1e-4 or cout <= 1e-4 or not (np.isfinite(cin) and np.isfinite(cout)):
            return 0.0
        return float(np.log(cout / cin))

    def evaluate_stages(
        self,
        q_in: float,
        ss_in: float,
        ss_primary: float,
        ss_out: float,
        bod_in: float,
        bod_out: float,
        cod_in: float,
        cod_out: float,
        sed_out: float = 0.5,
    ) -> PlantDiagnosticReport:
        """Analyze multi-stage removal efficiency and diagnose mechanical vs biological root causes."""
        # Global Removal efficiencies: 1 - Cout / Cin
        eff_bod = max(0.0, 1.0 - (bod_out / max(1e-4, bod_in))) if (np.isfinite(bod_in) and np.isfinite(bod_out)) else 0.85
        eff_cod = max(0.0, 1.0 - (cod_out / max(1e-4, cod_in))) if (np.isfinite(cod_in) and np.isfinite(cod_out)) else 0.80
        eff_ss = max(0.0, 1.0 - (ss_out / max(1e-4, ss_in))) if (np.isfinite(ss_in) and np.isfinite(ss_out)) else 0.85

        # Stage log ratios for Suspended Solids
        l_prim = self.log_ratio_efficiency(ss_in, ss_primary)  # Expected: approx -0.7 to -1.2
        l_sec = self.log_ratio_efficiency(ss_primary, ss_out)  # Expected: approx -1.5 to -2.5

        # Residuals relative to nominal baseline
        # Nominal: Primary removes ~60% (l ~ -0.9), Secondary removes ~85% (l ~ -1.9)
        res_prim = l_prim - (-0.90)
        res_sec = l_sec - (-1.90)

        # CUSUM accumulation on secondary settler degradation
        z_sec = max(0.0, res_sec)
        self.cusum_secondary = max(0.0, self.cusum_secondary + z_sec - self.cusum_k)

        alarms: List[str] = []
        if self.cusum_secondary > self.cusum_h:
            alarms.append(f"CUSUM Alarm: Secondary Clarifier Efficiency Deterioration (S={self.cusum_secondary:.2f})")

        # Extract diagnostic feature vector: [delta_Q, bio_impairment, ss_impairment, prim_residual, sed_s]
        delta_q = (q_in - 37000.0) / 10000.0 if np.isfinite(q_in) else 0.0
        bio_loss = max(0.0, (1.0 - eff_bod) - 0.15) * 5.0
        ss_loss = max(0.0, (1.0 - eff_ss) - 0.15) * 5.0

        obs_vector = np.array([
            max(0.0, delta_q),
            bio_loss,
            ss_loss,
            res_prim,
            sed_out if np.isfinite(sed_out) else 0.5,
        ])

        # Cosine similarity ignores magnitude, so on its own it matches an
        # ordinary day to whichever fault signature points the same way. A day
        # is only attributed to a fault once it has moved away from normal.
        deviation = float(np.linalg.norm(obs_vector))
        hypotheses: List[Tuple[str, float]] = []
        if deviation <= self.normal_radius:
            hypotheses.append(("Normal Operation", 1.0))
        else:
            for name, proto in self.FAULT_PROTOTYPES.items():
                similarity = float(np.dot(obs_vector, proto) / (deviation * np.linalg.norm(proto)))
                hypotheses.append((name, similarity))
            hypotheses.sort(key=lambda h: h[1], reverse=True)

        return PlantDiagnosticReport(
            global_bod_removal=eff_bod * 100.0,
            global_cod_removal=eff_cod * 100.0,
            global_ss_removal=eff_ss * 100.0,
            primary_efficiency_residual=res_prim,
            secondary_efficiency_residual=res_sec,
            active_alarms=alarms,
            deviation=deviation,
            fault_hypotheses=hypotheses[:3],
        )
