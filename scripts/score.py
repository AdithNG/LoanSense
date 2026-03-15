"""Score one loan application (deployment simulation)."""

import argparse
from pathlib import Path

import pandas as pd

from src.data.preprocess import preprocess_features
from src.models.predict import load_pipeline, predict, predict_proba

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    p = argparse.ArgumentParser(description="Score one loan application")
    p.add_argument("--income", type=float, required=True)
    p.add_argument("--debt", type=float, required=True)
    p.add_argument("--employment_years", type=int, required=True)
    p.add_argument("--credit_score", type=int, required=True)
    p.add_argument("--loan_amount", type=float, default=50_000)
    p.add_argument("--savings_balance", type=float, default=10_000)
    p.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    args = p.parse_args()

    row = pd.DataFrame([{
        "income": args.income,
        "debt": args.debt,
        "employment_years": args.employment_years,
        "credit_score": args.credit_score,
        "loan_amount": args.loan_amount,
        "savings_balance": args.savings_balance,
        "approved": 0,
    }])
    row = preprocess_features(row)
    model, feature_cols = load_pipeline(Path(args.model_dir))
    prob = predict_proba(model, feature_cols, row)[0]
    decision = predict(model, feature_cols, row)[0]
    print(f"Approval probability: {prob:.4f}")
    print(f"Decision: {'Approved' if decision == 1 else 'Denied'}")


if __name__ == "__main__":
    main()
