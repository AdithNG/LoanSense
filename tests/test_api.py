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
    data = r.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "llm_configured" in data
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["llm_configured"], bool)


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


# --- /score-and-email ---


@patch("src.api.main.get_pipeline")
def test_score_and_email_email_only(mock_get_pipeline, train_val_test):
    from src.models.train import train_model
    from src.data.preprocess import get_feature_columns_for_model
    train_df, val_df, _ = train_val_test
    model, _, _, feature_cols = train_model(train_df, val_df, algorithm="gradient_boosting", seed=42)
    mock_get_pipeline.return_value = (model, feature_cols)
    # Patch where completion is used (email imports it), so API sees the mock
    with patch("src.llm.email.completion") as mock_llm:
        mock_llm.return_value = "Dear Jane, Your loan has been processed. Loan Services Team"
        r = client.post(
            "/score-and-email",
            json={
                "applicant_name": "Jane Doe",
                "income": 50000,
                "debt": 10000,
                "employment_years": 5,
                "credit_score": 650,
                "run_agent_pipeline": False,
            },
        )
    assert r.status_code == 200
    data = r.json()
    assert "approval_probability" in data
    assert data["decision"] in ("approved", "denied")
    assert "reason" in data
    assert "email" in data
    assert isinstance(data["email"], str)
    assert "Jane" in data["email"] or "Loan" in data["email"]


@patch("src.api.main.get_pipeline")
def test_score_and_email_with_agent_pipeline(mock_get_pipeline, train_val_test):
    from src.models.train import train_model
    from src.data.preprocess import get_feature_columns_for_model
    train_df, val_df, _ = train_val_test
    model, _, _, feature_cols = train_model(train_df, val_df, algorithm="gradient_boosting", seed=42)
    mock_get_pipeline.return_value = (model, feature_cols)
    with patch("src.agents.pipeline.run_agent_pipeline") as mock_pipeline:
        from src.agents.pipeline import AgentPipelineResult
        mock_pipeline.return_value = AgentPipelineResult(
            email="Dear Jane, ...",
            bias_score=0.1,
            escalated=False,
            passed_tough_check=True,
            next_best_offer=None,
            final_email_sent=True,
        )
        r = client.post(
            "/score-and-email",
            json={
                "applicant_name": "Jane",
                "income": 30000,
                "debt": 15000,
                "employment_years": 2,
                "credit_score": 600,
                "run_agent_pipeline": True,
            },
        )
    assert r.status_code == 200
    data = r.json()
    assert "email" in data
    assert "bias_score" in data
    assert "escalated" in data
    assert "next_best_offer" in data
    assert data["bias_score"] == 0.1
    assert data["escalated"] is False


def test_score_and_email_validation_error():
    """Invalid payload (missing required field) returns 422."""
    r = client.post(
        "/score-and-email",
        json={
            "applicant_name": "Jane",
            "income": 50000,
            "debt": 10000,
            "employment_years": 5,
            # credit_score missing
        },
    )
    assert r.status_code == 422
