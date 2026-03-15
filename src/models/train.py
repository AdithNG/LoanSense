"""Train Gradient Boosting and Random Forest for loan approval (deterministic models)."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from src.data.preprocess import get_feature_columns_for_model, preprocess_features, prepare_splits


def train_model(
    df_train,
    df_val,
    algorithm: str = "gradient_boosting",
    seed: int = 42,
):
    """Train on train set. Use algorithm in ['gradient_boosting', 'random_forest']."""
    feature_cols = get_feature_columns_for_model()
    X_train = df_train[feature_cols]
    y_train = df_train["approved"]
    X_val = df_val[feature_cols]
    y_val = df_val["approved"]

    if algorithm == "gradient_boosting":
        # Sequential trees, better with noisy/inconsistent data (self-improving)
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=seed,
        )
    elif algorithm == "random_forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    model.fit(X_train, y_train)
    return model, X_val, y_val, feature_cols


def evaluate_model(model, X_val, y_val, X_test=None, y_test=None) -> dict:
    """Compute metrics. If test set provided, report test metrics (production simulation)."""
    metrics = {}
    for name, X, y in [("validation", X_val, y_val), ("test", X_test, y_test)]:
        if X is None or y is None:
            continue
        pred = model.predict(X)
        pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else pred
        metrics[name] = {
            "accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, pred_proba)) if len(np.unique(y)) > 1 else 0.0,
            "classification_report": classification_report(y, pred, zero_division=0),
        }
    return metrics


def save_pipeline(model, feature_cols: list, metrics: dict, path: str | Path) -> None:
    """Persist model and metadata for deployment."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, path / "pipeline.joblib")
    with open(path / "metrics.json", "w") as f:
        json.dump(
            {k: {kk: vv for kk, vv in v.items() if kk != "classification_report"}
                for k, v in metrics.items()},
            f,
            indent=2,
        )
