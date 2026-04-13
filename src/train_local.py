"""
train_local.py
--------------
Offline training script that uses only numpy + scipy (no sklearn/lgbm).
Implements:
  - Ridge Regression (via scipy.linalg.lstsq with L2 penalty)
  - Random Forest (via a pure-numpy bagged decision stump ensemble)
  - Gradient Boosting (via iterative residual regression)

This version is used when sklearn/lightgbm are not available.
In production, replace with the full train.py using sklearn + LightGBM.
"""

import json, os, pickle
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TARGET_COL = "clv_90d"
ID_COL     = "customer_id"
DROP_COLS  = [ID_COL, TARGET_COL, "true_segment"]

np.random.seed(42)

# ── helpers ───────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def robust_scale(X):
    """RobustScaler: subtract median, divide by IQR."""
    med = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    return (X - med) / iqr, med, iqr

def apply_scale(X, med, iqr):
    return (X - med) / iqr

# ── Ridge Regression ──────────────────────────────────────────────────────────

class RidgeRegressor:
    def __init__(self, alpha=10.0):
        self.alpha = alpha
        self.coef_ = None
        self._med = self._iqr = None

    def fit(self, X, y):
        Xs, self._med, self._iqr = robust_scale(X)
        n, p = Xs.shape
        A = Xs.T @ Xs + self.alpha * np.eye(p)
        b = Xs.T @ y
        self.coef_ = np.linalg.solve(A, b)
        self.intercept_ = np.mean(y) - Xs.mean(axis=0) @ self.coef_
        self.feature_importances_ = np.abs(self.coef_)
        return self

    def predict(self, X):
        Xs = apply_scale(X, self._med, self._iqr)
        return Xs @ self.coef_ + self.intercept_

# ── Simple Decision Stump ─────────────────────────────────────────────────────

class DecisionStump:
    """Single-level decision tree on one feature."""
    def fit(self, X, y, sample_weight=None):
        n, p = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n) / n
        best_loss = np.inf
        self.feature_ = self.threshold_ = self.left_ = self.right_ = None
        for f in range(p):
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2
            for t in thresholds:
                mask = X[:, f] <= t
                if mask.sum() < 2 or (~mask).sum() < 2:
                    continue
                l_mean = np.average(y[mask],  weights=sample_weight[mask])
                r_mean = np.average(y[~mask], weights=sample_weight[~mask])
                pred = np.where(mask, l_mean, r_mean)
                loss = np.average((y - pred) ** 2, weights=sample_weight)
                if loss < best_loss:
                    best_loss = self.feature_ = None  # reset marker
                    best_loss = loss
                    self.feature_ = f
                    self.threshold_ = t
                    self.left_ = l_mean
                    self.right_ = r_mean
        if self.feature_ is None:
            self.left_ = self.right_ = np.average(y, weights=sample_weight)
            self.feature_ = 0
            self.threshold_ = X[:, 0].mean()
        return self

    def predict(self, X):
        mask = X[:, self.feature_] <= self.threshold_
        return np.where(mask, self.left_, self.right_)

# ── Random Forest (bagged stumps) ─────────────────────────────────────────────

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_features=0.7, random_state=42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_  = []
        self.feat_indices_ = []
        self.feature_importances_ = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n, p = X.shape
        k = max(1, int(p * self.max_features))
        feat_counts = np.zeros(p)

        for _ in range(self.n_estimators):
            idx   = rng.choice(n, n, replace=True)
            feats = rng.choice(p, k, replace=False)
            Xb, yb = X[idx][:, feats], y[idx]
            stump = DecisionStump().fit(Xb, yb)
            self.estimators_.append(stump)
            self.feat_indices_.append(feats)
            feat_counts[feats] += 1

        self.feature_importances_ = feat_counts / feat_counts.sum()
        return self

    def predict(self, X):
        preds = np.column_stack([
            est.predict(X[:, fi])
            for est, fi in zip(self.estimators_, self.feat_indices_)
        ])
        return preds.mean(axis=1)

# ── Gradient Boosting ─────────────────────────────────────────────────────────

class GradientBoostingRegressor:
    def __init__(self, n_estimators=80, learning_rate=0.1, random_state=42):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.random_state  = random_state
        self.estimators_   = []
        self.base_pred_    = 0.0
        self.feature_importances_ = None

    def fit(self, X, y):
        rng  = np.random.default_rng(self.random_state)
        n, p = X.shape
        self.base_pred_ = y.mean()
        pred = np.full(n, self.base_pred_)
        feat_counts = np.zeros(p)

        for i in range(self.n_estimators):
            residuals = y - pred
            # Subsample 80 % of rows + 70 % of columns
            row_idx  = rng.choice(n, int(0.8 * n), replace=False)
            col_idx  = rng.choice(p, max(1, int(0.7 * p)), replace=False)
            Xs, ys   = X[row_idx][:, col_idx], residuals[row_idx]
            stump    = DecisionStump().fit(Xs, ys)
            self.estimators_.append((stump, col_idx))
            feat_counts[col_idx] += 1
            pred += self.learning_rate * stump.predict(X[:, col_idx])

        self.feature_importances_ = feat_counts / feat_counts.sum()
        return self

    def predict(self, X):
        pred = np.full(len(X), self.base_pred_)
        for stump, col_idx in self.estimators_:
            pred += self.learning_rate * stump.predict(X[:, col_idx])
        return pred

# ── k-Fold CV ─────────────────────────────────────────────────────────────────

def kfold_cv(model_cls, model_kwargs, X, y, k=5):
    n = len(X)
    fold_size = n // k
    rmses = []
    for i in range(k):
        val_start = i * fold_size
        val_end   = (i + 1) * fold_size if i < k - 1 else n
        val_idx   = np.arange(val_start, val_end)
        train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n)])
        m = model_cls(**model_kwargs)
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[val_idx])
        rmses.append(rmse(y[val_idx], pred))
    return np.mean(rmses), np.std(rmses)

# ── main ──────────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("CLV PREDICTION MODEL TRAINING")
    print("=" * 60)

    df = pd.read_csv("data/features.csv")
    drop = [c for c in DROP_COLS if c in df.columns]
    customer_ids = df[ID_COL] if ID_COL in df.columns else pd.Series(range(len(df)))
    y_raw = df[TARGET_COL].values.astype(float)
    X_raw = df.drop(columns=drop).values.astype(float)
    feature_names = list(df.drop(columns=drop).columns)

    # Fill NaN
    col_means = np.nanmean(X_raw, axis=0)
    inds = np.where(np.isnan(X_raw))
    X_raw[inds] = np.take(col_means, inds[1])

    print(f"Dataset: {X_raw.shape[0]:,} customers | {X_raw.shape[1]} features")

    # Train / test 80/20
    n = len(y_raw)
    test_size = int(0.2 * n)
    idx = np.random.permutation(n)
    test_idx, train_idx = idx[:test_size], idx[test_size:]

    X_train, X_test = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y_raw[train_idx], y_raw[test_idx]
    y_log_train = np.log1p(y_train)

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── CV comparison ────────────────────────────────────────────────────────
    print("\n── Cross-Validation (5-fold, log scale) ─────────────────────")
    models_config = [
        ("Ridge",           RidgeRegressor,          {"alpha": 10.0}),
        ("RandomForest",    RandomForestRegressor,   {"n_estimators": 60}),
        ("GradientBoosting",GradientBoostingRegressor,{"n_estimators": 60}),
    ]
    cv_results = {}
    for label, cls, kwargs in models_config:
        mu, std = kfold_cv(cls, kwargs, X_train, y_log_train, k=5)
        cv_results[label] = float(mu)
        print(f"  {label:<22} CV RMSE (log): {mu:.4f} ± {std:.4f}")

    # ── Train on full train set ───────────────────────────────────────────────
    print("\n── Training on Full Train Set ──────────────────────────────")
    ridge = RidgeRegressor(alpha=10.0).fit(X_train, y_log_train)
    rf    = RandomForestRegressor(n_estimators=80).fit(X_train, y_log_train)
    gb    = GradientBoostingRegressor(n_estimators=80).fit(X_train, y_log_train)

    # ── Test-set evaluation ───────────────────────────────────────────────────
    print("\n── Test-Set Evaluation (original $ scale) ──────────────────")
    test_metrics = {}
    trained_models = {"Ridge": ridge, "RandomForest": rf, "GradientBoosting": gb}

    for label, model in trained_models.items():
        pred_log = model.predict(X_test)
        pred     = np.expm1(pred_log).clip(0)
        test_metrics[label] = {
            "rmse": rmse(y_test, pred),
            "mae" : mae(y_test, pred),
        }
        print(f"  {label:<22} RMSE=${test_metrics[label]['rmse']:.2f}  MAE=${test_metrics[label]['mae']:.2f}")

    best_label = min(test_metrics, key=lambda k: test_metrics[k]["rmse"])
    best_model = trained_models[best_label]
    print(f"\n✓ Best model: {best_label}")

    # ── Feature importance ────────────────────────────────────────────────────
    fi = getattr(best_model, "feature_importances_", None)
    if fi is not None:
        fi_df = pd.DataFrame({"feature": feature_names, "importance": fi})
        fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
        fi_df.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
        print("\nTop-10 Features:")
        print(fi_df.head(10).to_string(index=False))

    # ── Save artefacts ────────────────────────────────────────────────────────
    with open(MODELS_DIR / "best_model.pkl",    "wb") as f: pickle.dump(best_model, f)
    with open(MODELS_DIR / "ridge_model.pkl",   "wb") as f: pickle.dump(ridge, f)
    with open(MODELS_DIR / "rf_model.pkl",      "wb") as f: pickle.dump(rf, f)
    with open(MODELS_DIR / "lgbm_model.pkl",    "wb") as f: pickle.dump(gb, f)
    with open(MODELS_DIR / "feature_names.pkl", "wb") as f: pickle.dump(feature_names, f)

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

    # Test predictions
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["y_true"]      = y_test
    test_df["y_pred_best"] = np.expm1(best_model.predict(X_test)).clip(0)
    test_df.to_csv(MODELS_DIR / "test_predictions.csv", index=False)

    print("\n✓ All artefacts saved to models/")
    print("=" * 60)

if __name__ == "__main__":
    train()
