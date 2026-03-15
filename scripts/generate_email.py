"""Level 2: Generate customer email from ML decision using LLM."""

import argparse
import os

from dotenv import load_dotenv
load_dotenv()

from src.llm.email import generate_customer_email


def main():
    p = argparse.ArgumentParser(description="Generate customer email from loan decision (LLM)")
    p.add_argument("--decision", choices=["approve", "deny", "approved", "denied"], required=True)
    p.add_argument("--applicant_name", type=str, default="Valued Customer")
    args = p.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in .env or environment.")
        return
    email = generate_customer_email(args.decision, args.applicant_name)
    print(email)


if __name__ == "__main__":
    main()
