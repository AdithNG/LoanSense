"""Level 3: Run full agent pipeline (email -> bias check -> escalate or next-best-offer)."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root so your key is used (overrides any system/shell OPENAI_API_KEY)
_load_env = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_load_env, override=True)

from src.agents.pipeline import run_agent_pipeline


def main():
    p = argparse.ArgumentParser(description="Run agent pipeline: email + bias detection + next-best-offer")
    p.add_argument("--decision", choices=["approve", "deny", "approved", "denied"], required=True)
    p.add_argument("--applicant_name", type=str, default="Valued Customer")
    p.add_argument("--no-next-best-offer", action="store_true", help="Skip next-best-offer for denied")
    args = p.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in .env or environment.")
        return
    result = run_agent_pipeline(
        args.decision,
        args.applicant_name,
        include_next_best_offer_on_deny=not args.no_next_best_offer,
    )
    print("Bias score:", result.bias_score)
    print("Escalated to human:", result.escalated)
    if result.passed_tough_check is not None:
        print("Passed tough check:", result.passed_tough_check)
    if result.next_best_offer:
        print("Next best offer:", result.next_best_offer)
    print("Final email sent:", result.final_email_sent)
    print("\n--- Email ---\n")
    print(result.email)


if __name__ == "__main__":
    main()
