"""Train loan approval model: 80/10/10 split, Gradient Boosting (or Random Forest), save pipeline."""

import argparse
from pathlib import Path

from src.data import load_loan_data, load_sample_data, preprocess_features, prepare_splits
from src.models.train import train_model, evaluate_model, save_pipeline

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    p = argparse.ArgumentParser(description="Train loan approval model")
    p.add_argument("--data", type=str, default="", help="Path to loan_data.csv (empty = generate sample)")
    p.add_argument("--algorithm", choices=["gradient_boosting", "random_forest"], default="gradient_boosting")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    args = p.parse_args()

    if args.data:
        df = load_loan_data(args.data)
    else:
        df = load_sample_data(seed=args.seed)
    df = preprocess_features(df)
    train_df, val_df, test_df = prepare_splits(
        df, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )

    model, X_val, y_val, feature_cols = train_model(
        train_df, val_df, algorithm=args.algorithm, seed=args.seed
    )
    X_test = test_df[feature_cols] if len(test_df) else None
    y_test = test_df["approved"] if len(test_df) else None
    metrics = evaluate_model(model, X_val, y_val, X_test, y_test)

    out_dir = Path(args.out_dir)
    save_pipeline(model, feature_cols, metrics, out_dir)
    print("Saved pipeline to", out_dir)
    print("Validation metrics:", metrics.get("validation", {}))
    if "test" in metrics:
        print("Test metrics (production simulation):", metrics["test"])


if __name__ == "__main__":
    main()
