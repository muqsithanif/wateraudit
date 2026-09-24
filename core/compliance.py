"""Regulatory effluent discharge compliance guard and continuous risk indexing."""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class ComplianceStatus:
    ecri_score: float                # Environmental Compliance Risk Index [0.0 to 1.0]
    alert_band: str                  # "GREEN", "WATCH", "ACT", "RED"
    violations: List[str]            # List of violated parameters
    parameter_probabilities: Dict[str, float]  # Probability of exceedance per parameter


class RegulatoryComplianceGuard:
    """Monitors wastewater effluent compliance against statutory discharge standards."""

    DEFAULT_LIMITS = {
        "DBO-S": 30.0,   # BOD limit: 30 mg/L
        "DQO-S": 125.0,  # COD limit: 125 mg/L
        "SS-S": 35.0,    # Suspended solids limit: 35 mg/L
    }

    PH_LIMIT_LOW = 6.5
    PH_LIMIT_HIGH = 8.5

    def __init__(self, limits: Optional[Dict[str, float]] = None):
        self.limits = limits or self.DEFAULT_LIMITS

    def evaluate_sample(
        self,
        predicted_exceed_probs: Dict[str, float],
        predicted_medians: Dict[str, float],
        ph_output: float = 7.5,
        actual_measurements: Optional[Dict[str, float]] = None,
    ) -> ComplianceStatus:
        """Evaluate multi-parameter effluent compliance and compute ECRI risk index."""
        violations: List[str] = []
        param_probs = dict(predicted_exceed_probs)

        # 1. pH Compliance check (instantaneous probe)
        if ph_output < self.PH_LIMIT_LOW or ph_output > self.PH_LIMIT_HIGH:
            ph_prob = 1.0
            violations.append(f"pH out of bounds ({ph_output:.2f} not in [{self.PH_LIMIT_LOW}, {self.PH_LIMIT_HIGH}])")
        else:
            # Smooth margin probability
            dev = max(self.PH_LIMIT_LOW - ph_output, ph_output - self.PH_LIMIT_HIGH)
            ph_prob = float(1.0 / (1.0 + np.exp(-dev * 5.0)))
        param_probs["PH-S"] = ph_prob

        # 2. Check actual laboratory measurements if available (ground truth)
        if actual_measurements is not None:
            for param, limit in self.limits.items():
                if param in actual_measurements and np.isfinite(actual_measurements[param]):
                    val = actual_measurements[param]
                    if val > limit:
                        violations.append(f"{param} exceeded legal limit ({val:.1f} > {limit:.1f} mg/L)")

        # 3. Compute continuous Environmental Compliance Risk Index (ECRI)
        # Joint exceedance probability: 1 - prod(1 - pi_j)
        probs_list = [np.clip(p, 0.0, 0.999) for p in param_probs.values()]
        joint_prob = 1.0 - float(np.prod([1.0 - p for p in probs_list]))

        # Expected fractional severity delta: sum max(0, (y_med - limit) / limit)
        severity_penalty = 0.0
        for param, limit in self.limits.items():
            if param in predicted_medians:
                med = predicted_medians[param]
                if med > limit:
                    delta = (med - limit) / limit
                    severity_penalty += delta

        # ECRI = 1 - (1 - joint_prob) * exp(-severity_penalty)
        ecri = 1.0 - (1.0 - joint_prob) * float(np.exp(-severity_penalty))
        ecri = float(np.clip(ecri, 0.0, 1.0))

        # Alert Band categorization
        if len(violations) > 0 or ecri >= 0.80:
            band = "RED"
        elif ecri >= 0.50:
            band = "ACT"
        elif ecri >= 0.20:
            band = "WATCH"
        else:
            band = "GREEN"

        return ComplianceStatus(
            ecri_score=ecri,
            alert_band=band,
            violations=violations,
            parameter_probabilities=param_probs,
        )
