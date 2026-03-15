"""Tests for unified LLM client (OpenAI/Anthropic with retries)."""

from unittest.mock import patch

import pytest

from src.llm.client import completion, _get_provider


def test_get_provider_default():
    with patch.dict("os.environ", {}, clear=False):
        assert _get_provider() == "openai"
    with patch.dict("os.environ", {"LLM_PROVIDER": "anthropic"}):
        assert _get_provider() == "anthropic"
    with patch.dict("os.environ", {"LLM_PROVIDER": "openai"}):
        assert _get_provider() == "openai"


def test_completion_returns_text_openai():
    with patch("src.llm.client._get_provider", return_value="openai"):
        with patch("src.llm.client._openai_completion") as mock_openai:
            mock_openai.return_value = "Hello, approved."
            result = completion("Say hello", event_name="test")
    assert result == "Hello, approved."
    mock_openai.assert_called_once()


def test_completion_returns_text_anthropic():
    with patch("src.llm.client._get_provider", return_value="anthropic"):
        with patch("src.llm.client._anthropic_completion") as mock_anthropic:
            mock_anthropic.return_value = "Hello from Claude."
            result = completion("Say hello", event_name="test")
    assert result == "Hello from Claude."
    mock_anthropic.assert_called_once()


def test_completion_retries_on_failure():
    with patch("src.llm.client._get_provider", return_value="openai"):
        with patch("src.llm.client.time.sleep"):
            with patch("src.llm.client._openai_completion") as mock_openai:
                mock_openai.side_effect = [Exception("rate limit"), Exception("again"), "Done"]
                result = completion("Hi", event_name="test")
    assert result == "Done"
    assert mock_openai.call_count == 3


def test_completion_raises_after_max_retries():
    with patch("src.llm.client._get_provider", return_value="openai"):
        with patch("src.llm.client._openai_completion") as mock_openai:
            mock_openai.side_effect = Exception("fail")
            with pytest.raises(Exception, match="fail"):
                completion("Hi", event_name="test")
    assert mock_openai.call_count == 3
