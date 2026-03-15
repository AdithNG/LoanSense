"""Load trained pipeline and score one or many applications (deployment)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.preprocess import preprocess_features


def load_pipeline(path: str | Path):
    """Load saved model and feature columns."""
    path = Path(path)
    data = joblib.load(path / "pipeline.joblib")
    return data["model"], data["feature_cols"]


def predict_proba(model, feature_cols: list, X: pd.DataFrame) -> np.ndarray:
    """Return probability of approval (class 1)."""
    X = X[feature_cols].fillna(0)
    return model.predict_proba(X)[:, 1]


def predict(model, feature_cols: list, X: pd.DataFrame) -> np.ndarray:
    """Return binary approve (1) / deny (0)."""
    return (predict_proba(model, feature_cols, X) >= 0.5).astype(int)
