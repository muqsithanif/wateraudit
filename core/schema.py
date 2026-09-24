"""UCI Water Treatment Plant dataset schema, attribute taxonomy, and leakage guard."""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np


# 38 attributes in UCI Water Treatment Plant dataset
UCI_ATTRIBUTES = [
    "Q-E",       # 1: Input flow
    "ZN-E",      # 2: Input Zinc
    "PH-E",      # 3: Input pH
    "DBO-E",     # 4: Input BOD
    "DQO-E",     # 5: Input COD
    "SS-E",      # 6: Input Suspended Solids
    "SSV-E",     # 7: Input Volatile Suspended Solids
    "SED-E",     # 8: Input Sediments
    "COND-E",    # 9: Input Conductivity
    "PH-P",      # 10: Primary settler pH
    "DBO-P",     # 11: Primary settler BOD
    "SS-P",      # 12: Primary settler Suspended Solids
    "SSV-P",     # 13: Primary settler Volatile Suspended Solids
    "SED-P",     # 14: Primary settler Sediments
    "COND-P",    # 15: Primary settler Conductivity
    "PH-D",      # 16: Secondary settler pH
    "DBO-D",     # 17: Secondary settler BOD
    "DQO-D",     # 18: Secondary settler COD
    "SS-D",      # 19: Secondary settler Suspended Solids
    "SSV-D",     # 20: Secondary settler Volatile Suspended Solids
    "SED-D",     # 21: Secondary settler Sediments
    "COND-D",    # 22: Secondary settler Conductivity
    "PH-S",      # 23: Output pH
    "DBO-S",     # 24: Output BOD (Target 1)
    "DQO-S",     # 25: Output COD (Target 2)
    "SS-S",      # 26: Output Suspended Solids (Target 3)
    "SSV-S",     # 27: Output Volatile Suspended Solids
    "SED-S",     # 28: Output Sediments
    "COND-S",    # 29: Output Conductivity
    "RD-DBO-P",  # 30: Primary BOD performance
    "RD-SS-P",   # 31: Primary SS performance
    "RD-SED-P",  # 32: Primary Sediments performance
    "RD-DBO-S",  # 33: Secondary BOD performance
    "RD-DQO-S",  # 34: Secondary COD performance
    "RD-DBO-G",  # 35: Global BOD performance
    "RD-DQO-G",  # 36: Global COD performance
    "RD-SS-G",   # 37: Global SS performance
    "RD-SED-G",  # 38: Global Sediments performance
]

TARGET_COLUMNS = ["DBO-S", "DQO-S", "SS-S"]

# Feature Tiers
TIER_0_ONLINE_PROBES = [
    "Q-E",
    "PH-E", "PH-P", "PH-D", "PH-S",
    "COND-E", "COND-P", "COND-D", "COND-S",
]

TIER_1_RAPID_TESTS = [
    "SED-E", "SED-P", "SED-D",
    "SSV-E", "SSV-P", "SSV-D",
    "ZN-E",
]

FORBIDDEN_LEAKAGE_COLUMNS = [
    "RD-DBO-P", "RD-SS-P", "RD-SED-P",
    "RD-DBO-S", "RD-DQO-S",
    "RD-DBO-G", "RD-DQO-G", "RD-SS-G", "RD-SED-G",
    "DBO-S", "DQO-S", "SS-S", "SSV-S", "SED-S",
]


class WWTPDatasetLoader:
    """Loads and sanitizes the authentic UCI Water Treatment Plant operational dataset."""

    @staticmethod
    def load_raw(data_path: Optional[str] = None) -> pd.DataFrame:
        """Read raw .data file, parse missing symbols '?' into np.nan, and enforce float dtypes."""
        if data_path is None:
            root_dir = Path(__file__).resolve().parent.parent
            data_path = str(root_dir / "data" / "water-treatment.data")

        # First column is Date/Day identifier, followed by 38 numerical features
        col_names = ["Date"] + UCI_ATTRIBUTES
        df = pd.read_csv(
            data_path,
            names=col_names,
            header=None,
            na_values=["?"],
        )

        # Convert attributes to float32
        for col in UCI_ATTRIBUTES:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

        return df

    @staticmethod
    def get_feature_matrix(
        df: pd.DataFrame,
        include_tier_1: bool = True,
        lag_days: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Construct leak-free feature matrix X and target DataFrame Y.

        Enforces:
        - Strict exclusion of RD-* columns.
        - Inclusion of Tier 0 online telemetry.
        - Optional inclusion of Tier 1 rapid settleability tests.
        - Rolling lags for temporal trends.
        """
        features: List[str] = list(TIER_0_ONLINE_PROBES)
        if include_tier_1:
            features.extend(TIER_1_RAPID_TESTS)

        # Verify no leakage column is in features
        for f in features:
            if f in FORBIDDEN_LEAKAGE_COLUMNS:
                raise ValueError(f"CRITICAL LEAKAGE: Forbidden column {f} found in feature set!")

        X = df[features].copy()

        # Add temporal features: Lags & Rolling Statistics
        for col in ["Q-E", "COND-E", "PH-E"]:
            X[f"{col}_lag1"] = X[col].shift(1)
            X[f"{col}_rolling_mean7"] = X[col].rolling(7, min_periods=1).mean()

        # Relative flow change: delta_Q
        rolling_median_q = X["Q-E"].rolling(7, min_periods=1).median()
        X["delta_Q"] = (X["Q-E"] / (rolling_median_q + 1e-4)) - 1.0

        Y = df[TARGET_COLUMNS].copy()

        return X, Y
