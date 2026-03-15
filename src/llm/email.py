"""Level 2: LLM generates customer email from ML decision (approve/deny). Adds probabilistic component."""

import os

from src.llm.client import completion


def generate_customer_email(
    decision: str,
    applicant_name: str,
    reason: str | None = None,
    model: str | None = None,
) -> str:
    """Generate a professional, non-discriminatory email to the customer. Optionally include the reason for the decision."""
    decision = decision.lower()
    if decision not in ("approve", "approved", "deny", "denied"):
        raise ValueError("decision must be approve/deny (or approved/denied)")

    approved = decision in ("approve", "approved")
    reason_line = f"\nThe decision was based on: {reason}." if reason else ""
    prompt = f"""You are a professional loan officer. Generate a short, respectful email to the customer named {applicant_name}.

The loan decision is: {"APPROVED" if approved else "DENIED"}.{reason_line}

Rules:
- Be professional and empathetic. No insults, no discrimination, no bias.
- If a reason was provided, explain the outcome in plain language (e.g. why they were denied or what supported approval); do not quote internal criteria verbatim.
- Do not mention race, gender, religion, or any protected characteristic.
- If denied, do not be harsh; suggest they can reapply or contact support.
- Keep the email to 2-4 sentences.
- Sign off as "Loan Services Team"."""

    return completion(
        prompt,
        model=model,
        temperature=0.3,
        event_name="email_generated",
    )
