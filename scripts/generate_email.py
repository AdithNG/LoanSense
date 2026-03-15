"""Level 2: Generate customer email from ML decision using LLM."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root so your key is used (overrides any system/shell OPENAI_API_KEY)
_load_env = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_load_env, override=True)

from src.llm.email import generate_customer_email


def main():
    p = argparse.ArgumentParser(description="Generate customer email from loan decision (LLM)")
    p.add_argument("--decision", choices=["approve", "deny", "approved", "denied"], required=True)
    p.add_argument("--applicant_name", type=str, default="Valued Customer")
    p.add_argument("--reason", type=str, default="", help="Reason for decision (e.g. from explain_decision)")
    args = p.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in .env or environment.")
        return
    email = generate_customer_email(args.decision, args.applicant_name, reason=args.reason or None)
    print(email)


if __name__ == "__main__":
    main()
