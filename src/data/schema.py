"""Schema and column names for loan approval data."""

TARGET = "approved"
FEATURE_COLUMNS = [
    "income",
    "debt",
    "employment_years",
    "credit_score",
    "loan_amount",
    "savings_balance",
]
# Optional: categoricals for feature engineering
CATEGORICAL_COLUMNS = []  # e.g. ["employment_type"]
