"""Integration test: full flow score -> generate email -> agent pipeline (all LLM mocked)."""

from unittest.mock import patch

import pytest

from src.data import load_sample_data, preprocess_features
from src.data.preprocess import prepare_splits
from src.models.train import train_model
from src.models.predict import predict, predict_proba, explain_decision
from src.llm.email import generate_customer_email
from src.agents.pipeline import run_agent_pipeline, AgentPipelineResult


@patch("src.llm.client.completion")
def test_full_flow_score_to_agent_pipeline(mock_completion):
    """Score application -> get reason -> generate email -> run agent pipeline; assert structure."""
    # 1. Train a real model (small data)
    df = load_sample_data(n=300, seed=42)
    df = preprocess_features(df)
    train_df, val_df, test_df = prepare_splits(df, 0.8, 0.1, 0.1, seed=42)
    model, _, _, feature_cols = train_model(train_df, val_df, algorithm="gradient_boosting", seed=42)

    # 2. Score one row
    row = test_df.head(1)
    prob = float(predict_proba(model, feature_cols, row)[0])
    decision_int = int(predict(model, feature_cols, row)[0])
    decision = "approved" if decision_int == 1 else "denied"
    reason = explain_decision(row, decision_int)

    assert 0 <= prob <= 1
    assert decision in ("approved", "denied")
    assert isinstance(reason, str)
    assert len(reason) > 0

    # 3. Mock LLM: email, bias score, bias score (strict), next-best-offer (for deny)
    mock_completion.side_effect = [
        "Dear Customer, Your application has been processed. Loan Services Team",
        "0.1",
        "0.2",
        "Consider our secured loan product.",
    ]

    # 4. Run agent pipeline with deny to exercise full flow (uses mocked completion)
    result = run_agent_pipeline(
        "deny",
        "Jane Doe",
        reason=reason,
        include_next_best_offer_on_deny=True,
    )

    assert isinstance(result, AgentPipelineResult)
    assert isinstance(result.email, str)
    assert isinstance(result.bias_score, (int, float))
    assert 0 <= result.bias_score <= 1
    assert isinstance(result.escalated, bool)
    assert result.final_email_sent in (True, False)
    if result.next_best_offer is not None:
        assert isinstance(result.next_best_offer, str)
