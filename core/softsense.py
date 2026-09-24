"""Online soft-sensing with Quantile Gradient Boosted Trees and Conformal Prediction."""
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import lightgbm as lgb


@dataclass
class ConformalInterval:
    lower: float
    median: float
    upper: float
    coverage_level: float
    exceedance_prob: float


class QuantileSoftSensor:
    """Predicts effluent wastewater parameters (BOD, COD, SS) with distribution-free conformal bounds."""

    def __init__(self, target_name: str, alpha: float = 0.10, random_state: int = 42):
        self.target_name = target_name
        self.alpha = alpha
        self.random_state = random_state

        self.models: Dict[float, lgb.LGBMRegressor] = {}
        self.conformal_q_hat: float = 0.0
        self.calibration_scores: np.ndarray = np.array([])
        self.is_calibrated: bool = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit three quantile gradient boosted models on log-transformed targets: log(1 + y)."""
        valid_mask = y_train.notna() & (y_train >= 0)
        X_clean = X_train[valid_mask]
        y_log = np.log1p(y_train[valid_mask].values)

        quantiles = [0.05, 0.50, 0.95]
        for q in quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=80,
                learning_rate=0.04,
                num_leaves=8,
                min_child_samples=15,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1,
            )
            model.fit(X_clean, y_log)
            self.models[q] = model

    def calibrate(self, X_cal: pd.DataFrame, y_cal: pd.Series) -> float:
        """Compute Conformalized Quantile Regression (CQR) adjustment on holdout calibration set."""
        valid_mask = y_cal.notna() & (y_cal >= 0)
        X_clean = X_cal[valid_mask]
        y_true_log = np.log1p(y_cal[valid_mask].values)
        n_c = len(y_true_log)

        if n_c < 10:
            self.conformal_q_hat = 0.10
            self.is_calibrated = True
            return self.conformal_q_hat

        # Predict raw quantiles
        q_low_log = self.models[0.05].predict(X_clean)
        q_high_log = self.models[0.95].predict(X_clean)

        # Conformity scores: s_i = max(q_low - y, y - q_high)
        scores = np.maximum(q_low_log - y_true_log, y_true_log - q_high_log)
        self.calibration_scores = scores

        # Order statistic at (1 - alpha) level: ceil((n + 1)(1 - alpha)) / n
        k = int(np.ceil((n_c + 1) * (1.0 - self.alpha)))
        k = min(n_c, max(1, k))
        # Ensure non-negative conformal expansion buffer
        self.conformal_q_hat = max(0.05, float(np.sort(scores)[k - 1]))
        self.is_calibrated = True
        return self.conformal_q_hat

    def predict_interval(
        self,
        X_query: pd.DataFrame,
        regulatory_limit: Optional[float] = None,
    ) -> List[ConformalInterval]:
        """Produce calibrated prediction intervals [y_low, y_high] and exceedance probabilities."""
        q05_log = self.models[0.05].predict(X_query)
        q50_log = self.models[0.50].predict(X_query)
        q95_log = self.models[0.95].predict(X_query)

        # Fix quantile crossing
        stacked = np.sort(np.column_stack([q05_log, q50_log, q95_log]), axis=1)
        q05_log, q50_log, q95_log = stacked[:, 0], stacked[:, 1], stacked[:, 2]

        # Apply conformal expansion
        c_low_log = q05_log - self.conformal_q_hat
        c_high_log = q95_log + self.conformal_q_hat

        # Inverse transform to original concentration units (mg/L): expm1(y)
        y_low = np.clip(np.expm1(c_low_log), 0.0, None)
        y_med = np.clip(np.expm1(q50_log), 0.0, None)
        y_high = np.clip(np.expm1(c_high_log), 0.0, None)

        intervals: List[ConformalInterval] = []
        for i in range(len(X_query)):
            # Estimate exceedance probability: P(Y > limit | x)
            if regulatory_limit is not None:
                limit_log = np.log1p(regulatory_limit)
                # Linear interpolation between lower, median, and upper quantiles
                if limit_log <= c_low_log[i]:
                    prob = 0.95
                elif limit_log >= c_high_log[i]:
                    prob = 0.05
                elif limit_log <= q50_log[i]:
                    denom = max(1e-4, q50_log[i] - c_low_log[i])
                    prob = 0.95 - 0.45 * ((limit_log - c_low_log[i]) / denom)
                else:
                    denom = max(1e-4, c_high_log[i] - q50_log[i])
                    prob = 0.50 - 0.45 * ((limit_log - q50_log[i]) / denom)
                prob = float(np.clip(prob, 0.01, 0.99))
            else:
                prob = 0.0

            intervals.append(
                ConformalInterval(
                    lower=float(y_low[i]),
                    median=float(y_med[i]),
                    upper=float(y_high[i]),
                    coverage_level=1.0 - self.alpha,
                    exceedance_prob=prob,
                )
            )

        return intervals
