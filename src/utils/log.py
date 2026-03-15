"""Structured logging for pipeline steps (anonymized, suitable for compliance/debugging)."""

import json
import logging
import os
from typing import Any

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL) if hasattr(logging, LOG_LEVEL) else logging.INFO
ENV = os.environ.get("ENV", "development")


def get_logger(name: str) -> logging.Logger:
    """Logger with level from LOG_LEVEL env."""
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        log.addHandler(handler)
    log.setLevel(LOG_LEVEL)
    return log


def log_llm_event(
    event: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    success: bool | None = None,
    error: str | None = None,
    bias_score: float | None = None,
    escalated: bool | None = None,
    **kwargs: Any,
) -> None:
    """Emit a structured log line for LLM/pipeline events (anonymized, no PII)."""
    payload = {"event": event, "env": ENV}
    if provider is not None:
        payload["provider"] = provider
    if model is not None:
        payload["model"] = model
    if success is not None:
        payload["success"] = success
    if error is not None:
        payload["error"] = error
    if bias_score is not None:
        payload["bias_score"] = round(bias_score, 4)
    if escalated is not None:
        payload["escalated"] = escalated
    payload.update(kwargs)
    msg = json.dumps(payload)
    log = logging.getLogger("loansense.pipeline")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(handler)
    log.setLevel(LOG_LEVEL)
    if error:
        log.warning(msg)
    else:
        log.info(msg)
