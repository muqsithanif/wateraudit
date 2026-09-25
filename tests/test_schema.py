"""Unit tests for dataset loading, typing, and leakage prevention."""
import numpy as np
import pandas as pd
import pytest
from core.schema import WWTPDatasetLoader, FORBIDDEN_LEAKAGE_COLUMNS, UCI_ATTRIBUTES


def test_dataset_loading_and_types():
    df = WWTPDatasetLoader.load_raw()
    assert len(df) == 527
    assert "Date" in df.columns
    # Check all 38 attributes exist
    for attr in UCI_ATTRIBUTES:
        assert attr in df.columns
        assert df[attr].dtype == np.float32


def test_leakage_guard_strictly_enforced():
    df = WWTPDatasetLoader.load_raw()
    X, Y = WWTPDatasetLoader.get_feature_matrix(df, include_tier_1=True)

    # Asserts NO forbidden column is present in feature matrix
    for col in X.columns:
        assert col not in FORBIDDEN_LEAKAGE_COLUMNS, f"Data leakage detected! Forbidden column {col} in X"

    # Verify targets
    assert list(Y.columns) == ["DBO-S", "DQO-S", "SS-S"]


def test_lagged_lab_features_only_use_results_already_available():
    # Same-day effluent values are the targets. The tier-2 features may only
    # carry results that are back from the lab by prediction time: one record
    # for COD and SS, five for the five-day BOD test.
    df = WWTPDatasetLoader.load_raw()
    X, _ = WWTPDatasetLoader.get_feature_matrix(df, include_tier_1=True, include_tier_2=True)
    for col, lag in [("DQO-S", 1), ("SS-S", 1), ("DBO-S", 5)]:
        feature = X[f"{col}_lag{lag}"]
        assert feature.iloc[:lag].isna().all()
        pd.testing.assert_series_equal(
            feature.iloc[lag:].reset_index(drop=True),
            df[col].iloc[:-lag].reset_index(drop=True),
            check_names=False,
        )
    assert not any(c in X.columns for c in ["DBO-S", "DQO-S", "SS-S"])


def test_records_are_returned_in_calendar_order():
    # The raw file is in scrambled monthly blocks; everything time-based
    # (lags, rolling means, the chronological split) depends on this sort.
    df = WWTPDatasetLoader.load_raw()
    assert df["Date"].is_monotonic_increasing
    assert df["Date"].iloc[0] == pd.Timestamp("1990-01-01")
    assert df["Date"].iloc[-1] == pd.Timestamp("1991-10-30")
