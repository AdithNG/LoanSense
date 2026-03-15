"""Tests for agent pipeline logic (mocked LLM/OpenAI)."""

from unittest.mock import patch

import pytest

from src.agents.bias import should_escalate, ESCALATE_THRESHOLD
from src.agents.pipeline import run_agent_pipeline, AgentPipelineResult


def test_should_escalate():
    assert should_escalate(0.7, 0.6) is True
    assert should_escalate(0.6, 0.6) is True
    assert should_escalate(0.5, 0.6) is False
    assert should_escalate(0.0, 0.6) is False


@patch("src.agents.pipeline.get_next_best_offer")
@patch("src.agents.pipeline.bias_score_email")
@patch("src.agents.pipeline.generate_customer_email")
def test_run_agent_pipeline_escalated_when_high_bias(mock_email, mock_bias, mock_offer):
    mock_email.return_value = "Dear Customer, ..."
    mock_bias.return_value = 0.9  # high risk -> escalate
    result = run_agent_pipeline("deny", "Jane", bias_threshold=0.6)
    assert result.escalated is True
    assert result.final_email_sent is False
    mock_offer.assert_not_called()


@patch("src.agents.pipeline.get_next_best_offer")
@patch("src.agents.pipeline.bias_score_email")
@patch("src.agents.pipeline.generate_customer_email")
def test_run_agent_pipeline_sent_when_low_bias_approve(mock_email, mock_bias, mock_offer):
    mock_email.return_value = "Dear Jane, Congratulations, approved."
    mock_bias.return_value = 0.1  # low risk
    result = run_agent_pipeline("approve", "Jane", bias_threshold=0.6)
    assert result.escalated is False
    assert result.final_email_sent is True
    assert result.next_best_offer is None
    mock_offer.assert_not_called()


@patch("src.agents.pipeline.get_next_best_offer")
@patch("src.agents.pipeline.bias_score_email")
@patch("src.agents.pipeline.generate_customer_email")
def test_run_agent_pipeline_deny_includes_next_best_offer(mock_email, mock_bias, mock_offer):
    mock_email.return_value = "Dear Jane, We cannot approve your loan."
    mock_bias.return_value = 0.1
    mock_offer.return_value = "Consider our secured loan product."
    result = run_agent_pipeline("deny", "Jane", bias_threshold=0.6, include_next_best_offer_on_deny=True)
    assert result.escalated is False
    assert result.final_email_sent is True
    assert result.next_best_offer == "Consider our secured loan product."
    assert "Consider our secured loan" in result.email
    mock_offer.assert_called_once()


@patch("src.agents.pipeline.get_next_best_offer")
@patch("src.agents.pipeline.bias_score_email")
@patch("src.agents.pipeline.generate_customer_email")
def test_run_agent_pipeline_tough_check_fails_escalates(mock_email, mock_bias, mock_offer):
    mock_email.return_value = "Dear Jane, ..."
    mock_bias.side_effect = [0.2, 0.8]  # first low, tougher agent returns high
    result = run_agent_pipeline("approve", "Jane", bias_threshold=0.6)
    assert result.escalated is True
    assert result.passed_tough_check is False
    assert result.final_email_sent is False
