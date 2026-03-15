"""Level 3: Bias/discrimination detection on generated email. Agent scores and decides to escalate or re-run."""

import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Threshold above which we escalate to human or send to stricter agent
ESCALATE_THRESHOLD = 0.6  # 0-1 risk score


def bias_score_email(email_text: str, model: str | None = None, strict: bool = False) -> float:
    """
    Assess email for bias, discrimination, or unprofessional content.
    Returns a score in [0, 1] where higher = more risky.
    """
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    strict_note = " Apply very strict criteria: any hint of bias, condescension, or insensitivity must be flagged." if strict else ""

    prompt = f"""You are a compliance agent. Score this customer email for risk of bias, discrimination, or unprofessional content.
{strict_note}

Email:
---
{email_text}
---

Reply with ONLY a single number between 0 and 1:
- 0 = completely safe, professional, no bias
- 1 = clearly biased, discriminatory, insulting, or inappropriate

Output only the number, no explanation."""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = (resp.choices[0].message.content or "0").strip()
    try:
        score = float(text)
    except ValueError:
        score = 0.5  # ambiguous response -> treat as medium risk
    return max(0.0, min(1.0, score))


def should_escalate(risk_score: float, threshold: float = ESCALATE_THRESHOLD) -> bool:
    """If score >= threshold, escalate to human or send to tougher agent."""
    return risk_score >= threshold
