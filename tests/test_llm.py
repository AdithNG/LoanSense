"""Tests for LLM email generation (mocked completion client)."""

from unittest.mock import patch

import pytest

from src.llm.email import generate_customer_email


def test_generate_customer_email_approve():
    with patch("src.llm.email.completion") as mock_completion:
        mock_completion.return_value = "Dear Customer, Your loan has been approved. Best, Loan Services Team"
        result = generate_customer_email("approve", "Jane Doe")
    assert "approved" in result.lower() or "approve" in result.lower()
    assert "Jane" in result or "Customer" in result


def test_generate_customer_email_deny():
    with patch("src.llm.email.completion") as mock_completion:
        mock_completion.return_value = "Dear Customer, We are unable to approve your loan. Please contact us. Loan Services Team"
        result = generate_customer_email("deny", "Bob")
    assert "Bob" in result or "Customer" in result


def test_generate_customer_email_approved_denied_aliases():
    with patch("src.llm.email.completion") as mock_completion:
        mock_completion.return_value = "Email body"
        generate_customer_email("approved", "X")
        generate_customer_email("denied", "Y")
    assert mock_completion.call_count == 2


def test_generate_customer_email_invalid_decision():
    with pytest.raises(ValueError, match="decision must be"):
        generate_customer_email("maybe", "Jane")


def test_generate_customer_email_with_reason():
    with patch("src.llm.email.completion") as mock_completion:
        mock_completion.return_value = "Dear Jane, We could not approve your loan due to your debt-to-income ratio. Loan Services Team"
        result = generate_customer_email("deny", "Jane", reason="debt-to-income ratio above our guideline")
    assert "Jane" in result or "Customer" in result
    call_args = mock_completion.call_args
    prompt = call_args[0][0] if call_args[0] else call_args.kwargs.get("prompt", "")
    assert "debt-to-income ratio" in prompt
