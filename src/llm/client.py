"""Unified LLM client: OpenAI or Anthropic with retries and structured logging."""

import logging
import os
import time
from typing import Literal

from src.utils.log import get_logger, log_llm_event

logger = get_logger(__name__)

Provider = Literal["openai", "anthropic"]
DEFAULT_PROVIDER: Provider = "openai"
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0


def _get_provider() -> Provider:
    raw = (os.environ.get("LLM_PROVIDER") or "openai").strip().lower()
    if raw in ("openai", "anthropic"):
        return raw
    return DEFAULT_PROVIDER


def _openai_completion(model: str, messages: list[dict], temperature: float = 0.3) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def _anthropic_completion(model: str, messages: list[dict], temperature: float = 0.3) -> str:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    # Anthropic messages: convert [{"role":"user","content":"..."}] to their format
    user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": user_content}],
        temperature=temperature,
    )
    text = ""
    if resp.content:
        for block in resp.content:
            if hasattr(block, "text"):
                text += block.text
    return text.strip()


def completion(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    event_name: str = "llm_completion",
) -> str:
    """
    Single completion with retries and backoff. Uses LLM_PROVIDER (openai | anthropic).
    Logs to structured logger and raises after MAX_RETRIES failures.
    """
    provider = _get_provider()
    model = model or (
        os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        if provider == "openai"
        else os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    )
    messages = [{"role": "user", "content": prompt}]
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if provider == "openai":
                out = _openai_completion(model, messages, temperature)
            else:
                out = _anthropic_completion(model, messages, temperature)
            log_llm_event(event_name, provider=provider, model=model, success=True)
            return out
        except Exception as e:
            last_error = e
            log_llm_event(event_name, provider=provider, model=model, success=False, error=str(e))
            if attempt < MAX_RETRIES - 1:
                sleep_time = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning("LLM attempt %s failed, retrying in %.1fs: %s", attempt + 1, sleep_time, e)
                time.sleep(sleep_time)
    raise last_error
