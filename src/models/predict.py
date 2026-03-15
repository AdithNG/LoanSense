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


# Guardrail thresholds: deny regardless of model when outside these bounds
DTI_DENY_THRESHOLD = 0.5   # debt/income above this -> auto-deny
CREDIT_MIN_DENY = 400      # credit score below this -> auto-deny


def apply_guardrails(row: pd.DataFrame) -> tuple[int | None, str | None]:
    """
    Apply rule-based guardrails so extreme cases are always denied (synthetic data
    never sees debt >> income, so the model can wrongly approve). Returns (decision_int, reason)
    if guardrail fires, else (None, None).
    """
    if row is None or len(row) == 0:
        return None, None
    r = row.iloc[0]
    if "dti_ratio" in r:
        dti = float(r.get("dti_ratio", 0) or 0)
        if dti > DTI_DENY_THRESHOLD:
            return 0, "debt-to-income ratio above our guideline"
    if "credit_score" in r:
        cs = int(r.get("credit_score", 0) or 0)
        if cs < CREDIT_MIN_DENY:
            return 0, "credit score below our threshold"
    return None, None


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
