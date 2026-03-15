"""
Download a public loan/credit dataset and save as data/loan_data.csv for training.

UCI Credit Approval: https://archive.ics.uci.edu/ml/datasets/credit+approval
- 690 rows, anonymized (columns A1–A16). We map numeric columns to LoanSense schema.
- Run from project root: python scripts/download_loan_data.py
"""

import sys
from pathlib import Path

import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "loan_data.csv"

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
# UCI columns (0-indexed): 1=A2 income-like, 2=A3 debt-like, 7=A8 numeric, 13=A14, 14=A15, 15=A16 class +/-
# We map to: income, debt, employment_years, credit_score, loan_amount, savings_balance, approved


def download_uci_credit(url: str = UCI_URL) -> pd.DataFrame:
    """Fetch UCI Credit Approval CSV and return raw DataFrame (no headers)."""
    df = pd.read_csv(url, header=None)
    # Drop rows with missing values (?)
    df = df.replace("?", pd.NA).dropna()
    return df


def map_uci_to_loansense(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map UCI Credit (A1–A16, anonymized) to LoanSense schema.
    - Col 1: continuous -> income (scale to dollars, e.g. * 1000)
    - Col 2: continuous -> debt (* 1000)
    - Col 7: continuous -> employment_years (scale and cap 0–40)
    - Col 10: categorical/numeric -> credit_score (encode or use col 7/8 scaled to 300–850)
    - Col 13: numeric string -> loan_amount (as int)
    - Col 14: numeric -> savings_balance
    - Col 15: + or - -> approved (1 or 0)
    """
    out = pd.DataFrame()
    # Income: col 1, scale to ~10k–80k range (UCI values ~10–60)
    out["income"] = pd.to_numeric(df[1], errors="coerce") * 1_500
    out["debt"] = pd.to_numeric(df[2], errors="coerce") * 1_000
    # Employment: col 7 (e.g. 0.04–15) -> years 0–40
    emp = pd.to_numeric(df[7], errors="coerce")
    out["employment_years"] = (emp.clip(0, 10) * 3).fillna(0).astype(int)
    # Loan amount: col 13 (e.g. "00202" -> 202)
    out["loan_amount"] = pd.to_numeric(df[13], errors="coerce").fillna(0).astype(int)
    # Savings: col 14
    out["savings_balance"] = pd.to_numeric(df[14], errors="coerce").fillna(0).astype(int)
    # Credit score: UCI anonymized; derive variety in 300–850 from other numeric cols
    out["credit_score"] = (300 + (out["loan_amount"] % 400) + out["employment_years"] * 5).clip(300, 850).astype(int)
    # Approved: col 15 is + or -
    out["approved"] = (df[15].str.strip() == "+").astype(int)
    # Drop any row that still has NaN
    out = out.dropna().astype({"income": int, "debt": int, "employment_years": int, "credit_score": int, "loan_amount": int, "savings_balance": int})
    return out


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading UCI Credit Approval dataset...")
    try:
        raw = download_uci_credit()
    except Exception as e:
        print(f"Download failed: {e}")
        print("You can manually download from:")
        print("  https://archive.ics.uci.edu/ml/datasets/credit+approval")
        print("  Save the data file as data/crx.data (no header, comma-separated).")
        sys.exit(1)
    print(f"Loaded {len(raw)} rows. Mapping to LoanSense schema...")
    df = map_uci_to_loansense(raw)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print("Train with: python scripts/train.py --data data/loan_data.csv")


if __name__ == "__main__":
    main()
