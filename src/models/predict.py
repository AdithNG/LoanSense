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


def explain_decision(row: pd.DataFrame, decision: int) -> str:
    """
    Return a short, non-sensitive reason for the decision (for the LLM to use).
    decision: 1 = approved, 0 = denied.
    """
    if row is None or len(row) == 0:
        return "Application review."
    r = row.iloc[0]
    reasons = []
    if "dti_ratio" in r:
        dti = r.get("dti_ratio", 0) or 0
        if decision == 0 and dti > 0.4:
            reasons.append("debt-to-income ratio above our guideline")
        elif decision == 1 and dti <= 0.35:
            reasons.append("favorable debt-to-income ratio")
    if "credit_score" in r:
        cs = r.get("credit_score", 0) or 0
        if decision == 0 and cs < 620:
            reasons.append("credit score below our threshold")
        elif decision == 1 and cs >= 680:
            reasons.append("strong credit profile")
    if "employment_years" in r:
        ey = r.get("employment_years", 0) or 0
        if decision == 0 and ey < 2:
            reasons.append("insufficient employment history")
        elif decision == 1 and ey >= 3:
            reasons.append("stable employment history")
    if not reasons:
        reasons.append("overall application assessment")
    return "; ".join(reasons)
