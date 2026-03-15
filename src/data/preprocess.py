"""Preprocessing and feature engineering for loan approval."""

import pandas as pd

from .schema import FEATURE_COLUMNS, TARGET


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features. Handles missing values and adds derived features."""
    out = df[FEATURE_COLUMNS + [TARGET]].copy()
    # Fill missing numeric with median (noisy/inconsistent data)
    for col in FEATURE_COLUMNS:
        if out[col].dtype in ("int64", "float64"):
            out[col] = out[col].fillna(out[col].median())
    # Derived: debt-to-income ratio (important for loan approval)
    out["dti_ratio"] = out["debt"] / (out["income"] + 1)
    # Cap extreme DTI
    out["dti_ratio"] = out["dti_ratio"].clip(0, 2)
    return out


def get_feature_columns_for_model() -> list[str]:
    """Feature set used for training (including engineered)."""
    return FEATURE_COLUMNS + ["dti_ratio"]


def prepare_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """Split into train / validation / test (e.g. 80/10/10) to avoid overfitting."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    return df.iloc[:t], df.iloc[t : t + v], df.iloc[t + v :]
