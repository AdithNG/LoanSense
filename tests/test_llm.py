"""Tests for LLM email generation (mocked OpenAI)."""

from unittest.mock import MagicMock, patch

import pytest

from src.llm.email import generate_customer_email


def test_generate_customer_email_approve():
    mock_content = MagicMock()
    mock_content.content = "Dear Customer, Your loan has been approved. Best, Loan Services Team"
    with patch("src.llm.email.client") as mock_client:
        mock_client.chat.completions.create.return_value.choices = [MagicMock(message=mock_content)]
        result = generate_customer_email("approve", "Jane Doe")
    assert "approved" in result.lower() or "approve" in result.lower()
    assert "Jane" in result or "Customer" in result


def test_generate_customer_email_deny():
    mock_content = MagicMock()
    mock_content.content = "Dear Customer, We are unable to approve your loan. Please contact us. Loan Services Team"
    with patch("src.llm.email.client") as mock_client:
        mock_client.chat.completions.create.return_value.choices = [MagicMock(message=mock_content)]
        result = generate_customer_email("deny", "Bob")
    assert "Bob" in result or "Customer" in result


def test_generate_customer_email_approved_denied_aliases():
    mock_content = MagicMock()
    mock_content.content = "Email body"
    with patch("src.llm.email.client") as mock_client:
        mock_client.chat.completions.create.return_value.choices = [MagicMock(message=mock_content)]
        generate_customer_email("approved", "X")
        generate_customer_email("denied", "Y")
    assert mock_client.chat.completions.create.call_count == 2


def test_generate_customer_email_invalid_decision():
    with pytest.raises(ValueError, match="decision must be"):
        generate_customer_email("maybe", "Jane")


def test_generate_customer_email_with_reason():
    mock_content = MagicMock()
    mock_content.content = "Dear Jane, We could not approve your loan due to your debt-to-income ratio. Loan Services Team"
    with patch("src.llm.email.client") as mock_client:
        mock_client.chat.completions.create.return_value.choices = [MagicMock(message=mock_content)]
        result = generate_customer_email("deny", "Jane", reason="debt-to-income ratio above our guideline")
    assert "Jane" in result or "Customer" in result
    call_args = mock_client.chat.completions.create.call_args
    assert "debt-to-income ratio" in call_args.kwargs["messages"][0]["content"]
