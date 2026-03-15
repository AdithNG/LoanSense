"""
Hyperparameter tuning for the loan approval model (Gradient Boosting or Random Forest).
Uses Optuna to maximize validation ROC-AUC. Run from project root:
  python scripts/tune.py [--data data/loan_data.csv] [--algorithm gradient_boosting|random_forest] [--trials 50]
"""

import argparse
from pathlib import Path

import optuna
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.data import load_loan_data, load_sample_data, preprocess_features, prepare_splits
from src.data.preprocess import get_feature_columns_for_model
from src.models.train import save_pipeline, evaluate_model

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def objective_gb(trial, X_train, y_train, X_val, y_val, feature_cols, seed):
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 2, 8)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, pred_proba)


def objective_rf(trial, X_train, y_train, X_val, y_val, feature_cols, seed):
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 4, 16)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, pred_proba)


def main():
    p = argparse.ArgumentParser(description="Tune loan approval model hyperparameters with Optuna")
    p.add_argument("--data", type=str, default="", help="Path to CSV (empty = sample data)")
    p.add_argument("--algorithm", choices=["gradient_boosting", "random_forest"], default="gradient_boosting")
    p.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    args = p.parse_args()

    if args.data:
        df = load_loan_data(args.data)
    else:
        df = load_sample_data(seed=args.seed)
    df = preprocess_features(df)
    train_df, val_df, test_df = prepare_splits(df, 0.8, 0.1, 0.1, seed=args.seed)
    feature_cols = get_feature_columns_for_model()
    X_train = train_df[feature_cols]
    y_train = train_df["approved"]
    X_val = val_df[feature_cols]
    y_val = val_df["approved"]

    if args.algorithm == "gradient_boosting":
        objective = lambda t: objective_gb(t, X_train, y_train, X_val, y_val, feature_cols, args.seed)
    else:
        objective = lambda t: objective_rf(t, X_train, y_train, X_val, y_val, feature_cols, args.seed)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("Best validation ROC-AUC:", study.best_value)
    print("Best params:", study.best_params)

    # Retrain best model and save
    best = study.best_params
    if args.algorithm == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=best["n_estimators"],
            max_depth=best["max_depth"],
            learning_rate=best["learning_rate"],
            random_state=args.seed,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=best["n_estimators"],
            max_depth=best["max_depth"],
            random_state=args.seed,
        )
    model.fit(X_train, y_train)
    X_test = test_df[feature_cols] if len(test_df) else None
    y_test = test_df["approved"] if len(test_df) else None
    metrics = evaluate_model(model, X_val, y_val, X_test, y_test)
    out_dir = Path(args.out_dir)
    save_pipeline(model, feature_cols, metrics, out_dir)
    print("Saved tuned pipeline to", out_dir)


if __name__ == "__main__":
    main()
