"""Level 3: Full agent pipeline: generate email -> bias check -> escalate or tougher agent -> optional next-best-offer."""

from dataclasses import dataclass
from typing import Literal

from src.llm.email import generate_customer_email
from src.agents.bias import bias_score_email, should_escalate, ESCALATE_THRESHOLD
from src.agents.next_best_offer import get_next_best_offer
from src.utils.log import log_llm_event


@dataclass
class AgentPipelineResult:
    email: str
    bias_score: float
    escalated: bool
    passed_tough_check: bool | None  # None if not escalated
    next_best_offer: str | None
    final_email_sent: bool  # False if escalated to human


def run_agent_pipeline(
    decision: Literal["approve", "deny", "approved", "denied"],
    applicant_name: str,
    *,
    reason: str | None = None,
    bias_threshold: float = ESCALATE_THRESHOLD,
    include_next_best_offer_on_deny: bool = True,
) -> AgentPipelineResult:
    """
    1. Generate email from ML decision (LLM), optionally including reason.
    2. Run bias detection agent -> score.
    3. If score >= threshold: escalate to human (we don't send; return escalated=True).
    4. Else: optionally run tougher bias check (strict=True), then if still ok, optionally
       add next-best-offer for deny and return final email.
    """
    log_llm_event("agent_pipeline_start", decision=decision.lower())
    email = generate_customer_email(decision, applicant_name, reason=reason)
    bias_score = bias_score_email(email)

    if should_escalate(bias_score, bias_threshold):
        log_llm_event("agent_pipeline_end", escalated=True, bias_score=bias_score)
        return AgentPipelineResult(
            email=email,
            bias_score=bias_score,
            escalated=True,
            passed_tough_check=None,
            next_best_offer=None,
            final_email_sent=False,
        )

    # Optional: tougher agent re-validates
    tough_score = bias_score_email(email, strict=True)
    passed_tough = not should_escalate(tough_score, bias_threshold)
    if not passed_tough:
        log_llm_event("agent_pipeline_end", escalated=True, bias_score=bias_score, passed_tough_check=False)
        return AgentPipelineResult(
            email=email,
            bias_score=bias_score,
            escalated=True,
            passed_tough_check=False,
            next_best_offer=None,
            final_email_sent=False,
        )

    next_best_offer = None
    denied = str(decision).lower() in ("deny", "denied")
    if denied and include_next_best_offer_on_deny:
        next_best_offer = get_next_best_offer(applicant_name, context="Loan was denied.")

    # Build final email (append next-best-offer if present)
    final_email = email
    if next_best_offer:
        final_email = f"{email}\n\nWe also have a recommendation for you: {next_best_offer}"

    log_llm_event("agent_pipeline_end", escalated=False, bias_score=bias_score, final_email_sent=True)
    return AgentPipelineResult(
        email=final_email,
        bias_score=bias_score,
        escalated=False,
        passed_tough_check=True,
        next_best_offer=next_best_offer if denied else None,
        final_email_sent=True,
    )
