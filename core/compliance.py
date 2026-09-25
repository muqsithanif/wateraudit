"""Effluent compliance check and a continuous risk index for the forecast."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import yaml


@dataclass
class ComplianceStatus:
    ecri_score: float                # Environmental Compliance Risk Index [0.0 to 1.0]
    alert_band: str                  # "GREEN", "WATCH", "ACT", "RED"
    violations: List[str]            # List of violated parameters
    parameter_probabilities: Dict[str, float]  # Probability of exceedance per parameter


class RegulatoryComplianceGuard:
    """Scores effluent against discharge limits loaded from configs/limits.yaml."""

    # Same values as configs/limits.yaml, used when no config is given.
    DEFAULT_LIMITS = {
        "DBO-S": 25.0,   # BOD, mg/L
        "DQO-S": 125.0,  # COD, mg/L
        "SS-S": 35.0,    # Suspended solids, mg/L
    }
    DEFAULT_PH_RANGE = (6.5, 8.5)
    DEFAULT_BANDS = (0.20, 0.50, 0.80)  # lower edges of WATCH, ACT, RED

    def __init__(
        self,
        limits: Optional[Dict[str, float]] = None,
        ph_range: Tuple[float, float] = DEFAULT_PH_RANGE,
        bands: Tuple[float, float, float] = DEFAULT_BANDS,
    ):
        self.limits = dict(limits or self.DEFAULT_LIMITS)
        self.ph_low, self.ph_high = ph_range
        self.watch, self.act, self.red = bands

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RegulatoryComplianceGuard":
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        standards = config["standards"]
        limits = {
            spec["column"]: float(spec["limit_max"])
            for name, spec in standards.items()
            if name != "pH"
        }
        ph = standards["pH"]
        bands = config["alert_bands"]
        return cls(
            limits=limits,
            ph_range=(float(ph["limit_min"]), float(ph["limit_max"])),
            bands=(float(bands["watch"]), float(bands["act"]), float(bands["red"])),
        )

    def evaluate_sample(
        self,
        predicted_exceed_probs: Dict[str, float],
        predicted_medians: Dict[str, float],
        ph_output: float = 7.5,
        actual_measurements: Optional[Dict[str, float]] = None,
    ) -> ComplianceStatus:
        """Combine predicted exceedance probabilities into one risk index.

        Pass `actual_measurements` only when reporting a day after the lab
        results are in. A forecast that is being scored must be evaluated
        without them, otherwise every measured breach is flagged by definition.
        """
        violations: List[str] = []
        param_probs = dict(predicted_exceed_probs)

        # 1. pH comes from an online probe, so it is a measurement, not a forecast
        if ph_output < self.ph_low or ph_output > self.ph_high:
            ph_prob = 1.0
            violations.append(f"pH out of bounds ({ph_output:.2f} not in [{self.ph_low}, {self.ph_high}])")
        else:
            # Smooth margin probability
            dev = max(self.ph_low - ph_output, ph_output - self.ph_high)
            ph_prob = float(1.0 / (1.0 + np.exp(-dev * 5.0)))
        param_probs["PH-S"] = ph_prob

        # 2. Lab results, when available
        if actual_measurements is not None:
            for param, limit in self.limits.items():
                if param in actual_measurements and np.isfinite(actual_measurements[param]):
                    val = actual_measurements[param]
                    if val > limit:
                        violations.append(f"{param} exceeded limit ({val:.1f} > {limit:.1f} mg/L)")

        # 3. Risk index: joint exceedance probability 1 - prod(1 - p_j),
        # pushed up further when a predicted median is already over its limit.
        probs_list = [np.clip(p, 0.0, 0.999) for p in param_probs.values()]
        joint_prob = 1.0 - float(np.prod([1.0 - p for p in probs_list]))

        severity_penalty = 0.0
        for param, limit in self.limits.items():
            if param in predicted_medians:
                med = predicted_medians[param]
                if med > limit:
                    severity_penalty += (med - limit) / limit

        ecri = 1.0 - (1.0 - joint_prob) * float(np.exp(-severity_penalty))
        ecri = float(np.clip(ecri, 0.0, 1.0))

        if len(violations) > 0 or ecri >= self.red:
            band = "RED"
        elif ecri >= self.act:
            band = "ACT"
        elif ecri >= self.watch:
            band = "WATCH"
        else:
            band = "GREEN"

        return ComplianceStatus(
            ecri_score=ecri,
            alert_band=band,
            violations=violations,
            parameter_probabilities=param_probs,
        )
