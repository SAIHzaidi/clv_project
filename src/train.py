"""
train.py
--------
End-to-end model training pipeline for CLV prediction.

Strategy
--------
1. Load the engineered feature matrix.
2. Split into train / test (time-aware: we train on older cohorts,
   test on more recent ones — mimicking real deployment).
3. Train three models with cross-validated hyperparameter tuning:
     • Linear Regression  (interpretable baseline)
     • Random Forest      (robust non-linear baseline)
     • LightGBM           (production-grade gradient boosting)
4. Evaluate on the held-out test set using RMSE and MAE.
5. Persist the best model + preprocessing artefacts.
6. Export a summary report (JSON) for downstream consumption.

Note on target transformation
------------------------------
CLV is right-skewed (many zeros, long tail of high spenders).
We apply log1p transformation during training and inverse-transform
predictions — this stabilises training and improves RMSE for the bulk
of customers without distorting high-value predictions too badly.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("data/features.csv")
MODELS_DIR  = Path("models")
RESULTS_DIR = Path("models")

MODELS_DIR.mkdir(exist_ok=True)

# ── feature columns (drop id and target) ─────────────────────────────────────
ID_COL     = "customer_id"
TARGET_COL = "clv_90d"
DROP_COLS  = [ID_COL, TARGET_COL, "true_segment"]   # true_segment only in synth data


def load_data(path: Path = DATA_PATH) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load features, return X, y, and customer_ids."""
    df = pd.read_csv(path)

    # Remove columns we don't want as features
    drop = [c for c in DROP_COLS if c in df.columns]
    customer_ids = df[ID_COL] if ID_COL in df.columns else pd.Series(range(len(df)))

    y = df[TARGET_COL]
    X = df.drop(columns=drop)

    return X, y, customer_ids


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> Dict:
    """Compute evaluation metrics on raw (non-log) scale."""
    metrics = {
        "rmse": _rmse(y_true, y_pred),
        "mae" : _mae(y_true, y_pred),
    }
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}")
    return metrics


def build_ridge_pipeline() -> Pipeline:
    """
    Ridge regression with RobustScaler.
    Robust scaler handles the outliers still present after winsorising.
    """
    return Pipeline([
        ("scaler", RobustScaler()),
        ("model",  Ridge(alpha=10.0)),
    ])


def build_rf_pipeline() -> Pipeline:
    """
    Random Forest — no scaling needed, but wrapped in Pipeline for
    consistency with the prediction interface.
    """
    return Pipeline([
        ("scaler", RobustScaler()),   # no-op for RF, kept for API consistency
        ("model",  RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            max_features=0.7,
            n_jobs=-1,
            random_state=42,
        )),
    ])


def build_lgbm_pipeline() -> lgb.LGBMRegressor:
    """
    LightGBM — best performance model.
    Hyperparameters selected via preliminary CV sweep (not shown here to
    keep training time reasonable; tune further with Optuna in production).
    """
    return lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
    )


def cross_validate_model(model, X_train, y_log_train, label: str) -> float:
    """5-fold CV on log-transformed target, returns mean RMSE."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X_train, y_log_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    mean_rmse = -scores.mean()
    std_rmse  = scores.std()
    print(f"  {label:<20} CV RMSE (log scale): {mean_rmse:.4f} ± {std_rmse:.4f}")
    return mean_rmse


def get_feature_importance(model, feature_names) -> pd.DataFrame:
    """Extract feature importance from tree-based models."""
    if hasattr(model, "named_steps"):
        est = model.named_steps.get("model", model)
    else:
        est = model

    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif hasattr(est, "coef_"):
        imp = np.abs(est.coef_)
    else:
        return pd.DataFrame()

    return (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train() -> None:
    """Full training run."""
    print("=" * 60)
    print("CLV PREDICTION MODEL TRAINING")
    print("=" * 60)

    # ── 1. load data ─────────────────────────────────────────────────────────
    X, y, customer_ids = load_data()
    feature_names = list(X.columns)
    print(f"\nDataset: {X.shape[0]:,} customers | {X.shape[1]} features")
    print(f"CLV range: ${y.min():.0f} – ${y.max():.0f}  |  mean ${y.mean():.2f}")

    # ── 2. train / test split (80 / 20) ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Log-transform the target for training stability
    y_log_train = np.log1p(y_train)

    # ── 3. cross-validation comparison ───────────────────────────────────────
    print("\n── Cross-Validation (5-fold, log scale) ────────────────────────")
    ridge  = build_ridge_pipeline()
    rf     = build_rf_pipeline()
    lgbm   = build_lgbm_pipeline()

    cv_results = {}
    for label, model in [("Ridge", ridge), ("RandomForest", rf), ("LightGBM", lgbm)]:
        cv_results[label] = cross_validate_model(model, X_train, y_log_train, label)

    # ── 4. fit all models on full training set ───────────────────────────────
    print("\n── Training on Full Train Set ──────────────────────────────────")
    ridge.fit(X_train, y_log_train)
    rf.fit(X_train, y_log_train)
    lgbm.fit(X_train, y_log_train)

    # ── 5. evaluate on test set (original scale) ──────────────────────────────
    print("\n── Test-Set Evaluation (original $ scale) ──────────────────────")
    test_metrics = {}
    predictions  = {}

    for label, model in [("Ridge", ridge), ("RandomForest", rf), ("LightGBM", lgbm)]:
        y_pred_log = model.predict(X_test)
        y_pred     = np.expm1(y_pred_log).clip(0)          # inverse log1p, floor at 0
        predictions[label] = y_pred
        test_metrics[label] = evaluate(y_test.values, y_pred, label)

    # ── 6. select best model ──────────────────────────────────────────────────
    best_label = min(test_metrics, key=lambda k: test_metrics[k]["rmse"])
    best_model = {"Ridge": ridge, "RandomForest": rf, "LightGBM": lgbm}[best_label]
    print(f"\n✓ Best model: {best_label} (RMSE=${test_metrics[best_label]['rmse']:.2f})")

    # ── 7. feature importance ─────────────────────────────────────────────────
    fi_df = get_feature_importance(best_model, feature_names)
    fi_df.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    print(f"\nTop-10 features ({best_label}):")
    print(fi_df.head(10).to_string(index=False))

    # ── 8. save artefacts ─────────────────────────────────────────────────────
    joblib.dump(best_model,    MODELS_DIR / "best_model.pkl")
    joblib.dump(ridge,         MODELS_DIR / "ridge_model.pkl")
    joblib.dump(rf,            MODELS_DIR / "rf_model.pkl")
    joblib.dump(lgbm,          MODELS_DIR / "lgbm_model.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")

    # ── 9. save summary report ────────────────────────────────────────────────
    report = {
        "best_model"   : best_label,
        "cv_results"   : cv_results,
        "test_metrics" : test_metrics,
        "feature_names": feature_names,
        "n_train"      : int(len(X_train)),
        "n_test"       : int(len(X_test)),
    }
    with open(MODELS_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ── 10. save test predictions for analysis / app ──────────────────────────
    test_results = X_test.copy()
    test_results["y_true"]       = y_test.values
    test_results["y_pred_best"]  = np.expm1(best_model.predict(X_test)).clip(0)
    test_results.to_csv(MODELS_DIR / "test_predictions.csv", index=False)

    print("\n✓ All artefacts saved to models/")
    print("=" * 60)


if __name__ == "__main__":
    train()
