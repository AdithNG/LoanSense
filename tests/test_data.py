"""Tests for data loading and preprocessing."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data import load_sample_data, load_loan_data, preprocess_features, prepare_splits
from src.data.schema import FEATURE_COLUMNS, TARGET
from src.data.preprocess import get_feature_columns_for_model


def test_load_sample_data_shape_and_columns():
    df = load_sample_data(n=100, seed=42)
    assert len(df) == 100
    assert set(FEATURE_COLUMNS) | {TARGET} <= set(df.columns)


def test_load_sample_data_approved_binary():
    df = load_sample_data(n=500, seed=42)
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_load_sample_data_reproducible():
    a = load_sample_data(n=50, seed=99)
    b = load_sample_data(n=50, seed=99)
    pd.testing.assert_frame_equal(a, b)


def test_preprocess_features_adds_dti():
    df = load_sample_data(n=100, seed=42)
    out = preprocess_features(df)
    assert "dti_ratio" in out.columns
    assert out["dti_ratio"].between(0, 2).all()


def test_preprocess_features_handles_missing():
    df = load_sample_data(n=200, seed=42)
    out = preprocess_features(df)
    assert out[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_get_feature_columns_for_model():
    cols = get_feature_columns_for_model()
    assert "dti_ratio" in cols
    assert all(c in cols for c in FEATURE_COLUMNS)


def test_prepare_splits_ratios():
    df = load_sample_data(n=1000, seed=42)
    df = preprocess_features(df)
    train, val, test = prepare_splits(df, 0.8, 0.1, 0.1, seed=42)
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100


def test_load_loan_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_loan_data(Path("/nonexistent/loan_data.csv"))


def test_load_loan_data_valid_csv(sample_df):
    path = Path(tempfile.mkdtemp()) / "loan_data.csv"
    try:
        sample_df[FEATURE_COLUMNS + [TARGET]].to_csv(path, index=False)
        loaded = load_loan_data(path)
        assert len(loaded) == len(sample_df)
        assert set(loaded.columns) >= set(FEATURE_COLUMNS) | {TARGET}
    finally:
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass


def test_load_loan_data_missing_columns():
    path = Path(tempfile.mkdtemp()) / "loan_data.csv"
    try:
        pd.DataFrame({"income": [1], "debt": [1]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="Missing columns"):
            load_loan_data(path)
    finally:
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass
