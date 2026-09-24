"""Unit tests for dataset loading, typing, and leakage prevention."""
import numpy as np
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
