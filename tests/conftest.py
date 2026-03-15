"""Pytest fixtures and config."""

import sys
from pathlib import Path

import pytest

# Ensure project root is on path when running pytest from repo root or tests/
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


@pytest.fixture
def sample_df():
    """Small preprocessed dataframe for model tests."""
    from src.data import load_sample_data, preprocess_features
    df = load_sample_data(n=200, seed=123)
    return preprocess_features(df)


@pytest.fixture
def train_val_test(sample_df):
    """Train, validation, test splits."""
    from src.data.preprocess import prepare_splits
    return prepare_splits(sample_df, 0.8, 0.1, 0.1, seed=123)
