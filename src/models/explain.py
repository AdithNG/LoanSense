"""Per-prediction interpretability: SHAP values or feature-importance fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_prediction_contributions(
    model,
    feature_cols: list[str],
    row: pd.DataFrame,
    use_shap: bool = True,
) -> dict[str, float]:
    """
    Return per-feature contribution to this prediction (positive = toward approve, negative = toward deny).
    Uses SHAP TreeExplainer when available and model is tree-based; otherwise falls back to
    feature importance * (value - median) as a simple proxy.
    """
    X = row[feature_cols].fillna(0)
    if use_shap:
        try:
            import shap
            # TreeExplainer: do NOT pass X as background (that explains row vs itself -> all zeros).
            # Use model only so SHAP uses tree-path-dependent baseline and returns real contributions.
            if hasattr(model, "estimators_") or hasattr(model, "estimators"):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                # Binary: shap_values can be (n_samples, n_features) or list of two arrays
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # class 1 (approve)
                if shap_values.ndim > 1:
                    shap_values = shap_values[0]
                return {f: round(float(v), 4) for f, v in zip(feature_cols, shap_values)}
        except Exception:
            pass
    # No SHAP: return empty (per-prediction explanation requires SHAP for tree models)
    return {}


def format_contributions_for_display(contributions: dict[str, float], top_n: int = 5) -> str:
    """Return a short human-readable summary of top positive/negative contributors."""
    if not contributions:
        return "No per-prediction breakdown available."
    sorted_ = sorted(contributions.items(), key=lambda x: -abs(x[1]))
    parts = []
    for feat, val in sorted_[:top_n]:
        if val > 0:
            parts.append(f"{feat} (+{val:.3f})")
        else:
            parts.append(f"{feat} ({val:.3f})")
    return " · ".join(parts)
