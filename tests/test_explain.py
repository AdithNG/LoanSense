"""Tests for model interpretability (SHAP contributions)."""

import pandas as pd

from src.models.explain import format_contributions_for_display, get_prediction_contributions


def test_format_contributions_empty():
    assert "No per-prediction" in format_contributions_for_display({})


def test_format_contributions_display():
    contrib = {"dti_ratio": -0.2, "credit_score": 0.15, "income": 0.05}
    out = format_contributions_for_display(contrib, top_n=2)
    assert "dti_ratio" in out
    assert "credit_score" in out
    assert "+" in out or "0.15" in out


def test_get_prediction_contributions_returns_dict(train_val_test):
    from src.models.train import train_model
    train_df, val_df, _ = train_val_test
    model, _, _, feature_cols = train_model(train_df, val_df, algorithm="gradient_boosting", seed=42)
    row = val_df.head(1)
    result = get_prediction_contributions(model, feature_cols, row)
    assert isinstance(result, dict)
    if result:
        for k, v in result.items():
            assert k in feature_cols
            assert isinstance(v, (int, float))
