"""Level 3: Next-best-offer agent for denied applicants (e.g. recommend alternative product)."""

import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_next_best_offer(
    applicant_name: str,
    context: str = "Customer was denied for the requested loan.",
    model: str | None = None,
) -> str:
    """
    Recommend a next-best offer (e.g. smaller loan, different product) to keep customer engaged.
    """
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    prompt = f"""You are a financial product recommender. Given this context:
{context}

Suggest one short, specific next-best offer for {applicant_name} (e.g. smaller loan amount, secured loan, or savings product). Keep it to 1-2 sentences, professional and helpful. Do not be discriminatory or biased."""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return (resp.choices[0].message.content or "").strip()
