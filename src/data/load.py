"""Load loan data from CSV or generate sample data."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import FEATURE_COLUMNS, TARGET


def load_loan_data(path: str | Path) -> pd.DataFrame:
    """Load loan data from CSV. Expects columns in schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    required = set(FEATURE_COLUMNS) | {TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Expected {required}.")
    return df


def load_sample_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic loan data for experimentation (inconsistent income/docs, noise)."""
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10.5, 0.6, n).astype(int)
    # Mix: most have debt 5–50% of income; ~15% have high debt (up to 2x income) so model sees deny cases
    high_debt_frac = 0.15
    n_high = int(n * high_debt_frac)
    debt_low = (income[: n - n_high] * rng.uniform(0.05, 0.5, n - n_high)).astype(int)
    debt_high = (income[n - n_high :] * rng.uniform(0.6, 2.0, n_high)).astype(int)
    debt = np.concatenate([debt_low, debt_high])
    employment_years = rng.integers(0, 35, n)
    credit_score = rng.integers(300, 850, n)
    loan_amount = (income * rng.uniform(0.5, 3.0, n)).astype(int)
    savings_balance = (income * rng.uniform(0, 2.0, n)).astype(int)
    # Approve/deny: ratio > 2.5, credit > 580, employment >= 1 (high-debt cases will usually deny)
    ratio = income / (debt + 1)
    approved = (
        (ratio > 2.5) & (credit_score > 580) & (employment_years >= 1)
    ).astype(int)
    # Add noise: flip ~5% of labels
    flip = rng.random(n) < 0.05
    approved = np.where(flip, 1 - approved, approved)
    df = pd.DataFrame({
        "income": income,
        "debt": debt,
        "employment_years": employment_years,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "savings_balance": savings_balance,
        TARGET: approved,
    })
    # Missing values in a few rows (inconsistent documentation)
    for col in ["credit_score", "employment_years"]:
        idx = rng.choice(n, size=int(n * 0.02), replace=False)
        df.loc[df.index[idx], col] = np.nan
    return df
