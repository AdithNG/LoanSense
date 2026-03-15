"""Tests for FastAPI endpoints."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_score_requires_trained_model():
    # If models/ doesn't have pipeline.joblib, /score returns 503
    r = client.post(
        "/score",
        json={
            "income": 50000,
            "debt": 10000,
            "employment_years": 5,
            "credit_score": 650,
        },
    )
    # Either 200 (model exists) or 503 (not trained)
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        data = r.json()
        assert "approval_probability" in data
        assert data["decision"] in ("approved", "denied")
        assert "reason" in data


@patch("src.api.main.get_pipeline")
def test_score_returns_decision(mock_get_pipeline, train_val_test):
    from src.models.train import train_model
    from src.models.predict import predict_proba, predict
    from src.data.preprocess import get_feature_columns_for_model
    train_df, val_df, _ = train_val_test
    model, _, _, feature_cols = train_model(train_df, val_df, algorithm="gradient_boosting", seed=42)
    mock_get_pipeline.return_value = (model, feature_cols)
    r = client.post(
        "/score",
        json={
            "income": 50000,
            "debt": 10000,
            "employment_years": 5,
            "credit_score": 650,
            "loan_amount": 50000,
            "savings_balance": 10000,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "approval_probability" in data
    assert "reason" in data
    assert data["decision"] in ("approved", "denied")
    assert 0 <= data["approval_probability"] <= 1
