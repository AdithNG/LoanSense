"""Tests for model training and prediction."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.models.train import train_model, evaluate_model, save_pipeline
from src.models.predict import load_pipeline, predict, predict_proba, explain_decision, apply_guardrails
from src.data.preprocess import get_feature_columns_for_model


def test_train_model_gradient_boosting(train_val_test):
    train_df, val_df, _ = train_val_test
    model, X_val, y_val, feature_cols = train_model(train_df, val_df, algorithm="gradient_boosting", seed=42)
    assert model is not None
    assert list(X_val.columns) == feature_cols
    assert len(y_val) == len(X_val)


def test_train_model_random_forest(train_val_test):
    train_df, val_df, _ = train_val_test
    model, X_val, y_val, feature_cols = train_model(train_df, val_df, algorithm="random_forest", seed=42)
    assert model is not None
    assert "randomforest" in type(model).__name__.lower()


def test_train_model_invalid_algorithm(train_val_test):
    train_df, val_df, _ = train_val_test
    with pytest.raises(ValueError, match="Unknown algorithm"):
        train_model(train_df, val_df, algorithm="invalid", seed=42)


def test_evaluate_model_returns_metrics(train_val_test):
    train_df, val_df, test_df = train_val_test
    feature_cols = get_feature_columns_for_model()
    model, X_val, y_val, _ = train_model(train_df, val_df, algorithm="gradient_boosting", seed=42)
    X_test = test_df[feature_cols]
    y_test = test_df["approved"]
    metrics = evaluate_model(model, X_val, y_val, X_test, y_test)
    assert "validation" in metrics
    assert "test" in metrics
    for name in ["validation", "test"]:
        assert "accuracy" in metrics[name]
        assert "f1" in metrics[name]
        assert "roc_auc" in metrics[name]
        assert 0 <= metrics[name]["accuracy"] <= 1


def test_predict_proba_shape(train_val_test):
    train_df, val_df, test_df = train_val_test
    model, _, _, feature_cols = train_model(train_df, val_df, algorithm="gradient_boosting", seed=42)
    proba = predict_proba(model, feature_cols, test_df)
    assert proba.shape == (len(test_df),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_predict_binary(train_val_test):
    train_df, val_df, test_df = train_val_test
    model, _, _, feature_cols = train_model(train_df, val_df, algorithm="gradient_boosting", seed=42)
    pred = predict(model, feature_cols, test_df)
    assert set(pred).issubset({0, 1})
    assert len(pred) == len(test_df)


def test_save_and_load_pipeline(train_val_test, tmp_path):
    train_df, val_df, _ = train_val_test
    model, X_val, y_val, feature_cols = train_model(train_df, val_df, algorithm="gradient_boosting", seed=42)
    metrics = evaluate_model(model, X_val, y_val)
    save_pipeline(model, feature_cols, metrics, tmp_path)
    assert (tmp_path / "pipeline.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    loaded_model, loaded_cols = load_pipeline(tmp_path)
    assert loaded_cols == feature_cols
    proba = predict_proba(loaded_model, loaded_cols, train_df.head(1))
    assert proba.shape == (1,)
    with open(tmp_path / "metrics.json") as f:
        m = json.load(f)
    assert "validation" in m


def test_apply_guardrails_high_dti():
    row = pd.DataFrame([{"dti_ratio": 0.6, "credit_score": 600}])
    decision, reason = apply_guardrails(row)
    assert decision == 0
    assert "debt-to-income" in (reason or "")


def test_apply_guardrails_low_credit():
    row = pd.DataFrame([{"dti_ratio": 0.3, "credit_score": 350}])
    decision, reason = apply_guardrails(row)
    assert decision == 0
    assert "credit" in (reason or "").lower()


def test_apply_guardrails_passes():
    row = pd.DataFrame([{"dti_ratio": 0.35, "credit_score": 650}])
    decision, reason = apply_guardrails(row)
    assert decision is None
    assert reason is None


def test_explain_decision_denied_high_dti():
    row = pd.DataFrame([{
        "income": 50_000, "debt": 30_000, "employment_years": 1,
        "credit_score": 500, "dti_ratio": 0.6,
    }])
    reason = explain_decision(row, 0)
    assert "debt-to-income" in reason or "application" in reason.lower()


def test_explain_decision_approved():
    row = pd.DataFrame([{
        "income": 80_000, "debt": 10_000, "employment_years": 5,
        "credit_score": 720, "dti_ratio": 0.125,
    }])
    reason = explain_decision(row, 1)
    assert len(reason) > 0
