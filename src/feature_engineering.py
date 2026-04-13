"""
feature_engineering.py
-----------------------
Transforms raw transactional data into a rich feature matrix
suitable for CLV prediction.

Methodology
-----------
We adopt the classic RFM framework (Recency, Frequency, Monetary value)
and extend it with behavioural and trend features that capture:
  - Velocity of spending (is the customer accelerating or decelerating?)
  - Order value consistency (predictable vs. erratic buyers)
  - Tenure effects (new vs. established customers)
  - Seasonal interaction (holiday buyers)

All features are computed with respect to a configurable observation
cut-off date so the same pipeline can be used for both back-testing
and live scoring.

Target variable
---------------
CLV = total spend in the 90-day window AFTER the cut-off date.
Customers with no activity in the target window get CLV = 0.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


# ── constants ─────────────────────────────────────────────────────────────────
OBSERVATION_PERIOD_DAYS = 365   # history window used to build features
TARGET_PERIOD_DAYS      = 90    # future window we want to predict


def load_transactions(path: str = "data/transactions.csv") -> pd.DataFrame:
    """Load and lightly clean the raw transaction file."""
    df = pd.read_csv(path, parse_dates=["transaction_date"])

    # Basic sanity checks
    df = df[df["purchase_amount"] > 0].copy()
    df = df.dropna(subset=["customer_id", "transaction_date", "purchase_amount"])
    df["customer_id"] = df["customer_id"].astype(str)

    return df


def _set_cut_off(
    df: pd.DataFrame,
    cut_off: Optional[pd.Timestamp],
) -> pd.Timestamp:
    """If no cut-off supplied, place it so we still have a full target window."""
    if cut_off is not None:
        return pd.Timestamp(cut_off)
    max_date = df["transaction_date"].max()
    return max_date - pd.Timedelta(days=TARGET_PERIOD_DAYS)


def compute_rfm_features(
    df: pd.DataFrame,
    cut_off: pd.Timestamp,
) -> pd.DataFrame:
    """
    Core RFM features computed over the observation window.

    Recency  – days since the customer last purchased (lower = more recent)
    Frequency – number of distinct purchase events
    Monetary  – total spend
    """
    obs = df[
        (df["transaction_date"] <= cut_off)
        & (df["transaction_date"] > cut_off - pd.Timedelta(days=OBSERVATION_PERIOD_DAYS))
    ].copy()

    rfm = (
        obs.groupby("customer_id")
        .agg(
            last_purchase_date=("transaction_date", "max"),
            first_purchase_date=("transaction_date", "min"),
            frequency=("transaction_date", "count"),
            monetary=("purchase_amount", "sum"),
        )
        .reset_index()
    )

    rfm["recency"]        = (cut_off - rfm["last_purchase_date"]).dt.days
    rfm["customer_tenure"] = (cut_off - rfm["first_purchase_date"]).dt.days

    return rfm


def compute_advanced_features(
    df: pd.DataFrame,
    rfm: pd.DataFrame,
    cut_off: pd.Timestamp,
) -> pd.DataFrame:
    """
    Extend RFM with order-level and temporal behavioural features.

    avg_order_value        – mean spend per transaction
    std_order_value        – spend volatility (predictable vs. erratic)
    avg_days_between_txns  – inter-purchase time (purchase cadence)
    cv_order_value         – coefficient of variation (std / mean)
    recent_spend_90d       – spend in last 90 days of obs. window
    historical_spend_rest  – spend in the remaining part of obs. window
    spend_trend            – log ratio: recent / historical (growth signal)
    purchase_rate          – purchases per active day
    max_order_value        – largest single purchase (whale signal)
    months_with_purchase   – breadth of purchasing across calendar months
    """
    obs = df[
        (df["transaction_date"] <= cut_off)
        & (df["transaction_date"] > cut_off - pd.Timedelta(days=OBSERVATION_PERIOD_DAYS))
    ].copy()

    # ── per-customer order-level stats ────────────────────────────────────────
    order_stats = (
        obs.groupby("customer_id")["purchase_amount"]
        .agg(
            avg_order_value="mean",
            std_order_value="std",
            max_order_value="max",
            min_order_value="min",
        )
        .reset_index()
    )
    order_stats["std_order_value"] = order_stats["std_order_value"].fillna(0)
    order_stats["cv_order_value"]  = (
        order_stats["std_order_value"] / order_stats["avg_order_value"].replace(0, np.nan)
    ).fillna(0)

    # ── inter-purchase gap ────────────────────────────────────────────────────
    def _mean_gap(group: pd.DataFrame) -> float:
        dates = group["transaction_date"].sort_values()
        if len(dates) < 2:
            return np.nan
        return dates.diff().dt.days.dropna().mean()

    gap_df = (
        obs.groupby("customer_id")
        .apply(_mean_gap)
        .reset_index()
        .rename(columns={0: "avg_days_between_txns"})
    )

    # ── spend trend: recent 90d vs rest of obs window ─────────────────────────
    recent_cut = cut_off - pd.Timedelta(days=90)

    recent_spend = (
        obs[obs["transaction_date"] > recent_cut]
        .groupby("customer_id")["purchase_amount"]
        .sum()
        .reset_index()
        .rename(columns={"purchase_amount": "recent_spend_90d"})
    )

    historical_spend = (
        obs[obs["transaction_date"] <= recent_cut]
        .groupby("customer_id")["purchase_amount"]
        .sum()
        .reset_index()
        .rename(columns={"purchase_amount": "historical_spend_rest"})
    )

    # ── purchase rate ─────────────────────────────────────────────────────────
    def _purchase_rate(group: pd.DataFrame) -> float:
        tenure = (group["transaction_date"].max() - group["transaction_date"].min()).days
        return len(group) / max(1, tenure)

    rate_df = (
        obs.groupby("customer_id")
        .apply(_purchase_rate)
        .reset_index()
        .rename(columns={0: "purchase_rate"})
    )

    # ── calendar breadth ──────────────────────────────────────────────────────
    obs["year_month"] = obs["transaction_date"].dt.to_period("M")
    breadth_df = (
        obs.groupby("customer_id")["year_month"]
        .nunique()
        .reset_index()
        .rename(columns={"year_month": "months_with_purchase"})
    )

    # ── merge everything ──────────────────────────────────────────────────────
    advanced = (
        rfm
        .merge(order_stats,       on="customer_id", how="left")
        .merge(gap_df,            on="customer_id", how="left")
        .merge(recent_spend,      on="customer_id", how="left")
        .merge(historical_spend,  on="customer_id", how="left")
        .merge(rate_df,           on="customer_id", how="left")
        .merge(breadth_df,        on="customer_id", how="left")
    )

    advanced["recent_spend_90d"]      = advanced["recent_spend_90d"].fillna(0)
    advanced["historical_spend_rest"] = advanced["historical_spend_rest"].fillna(0)

    # Spend trend: log ratio — clamped to avoid ±inf
    eps = 1e-3
    advanced["spend_trend"] = np.log(
        (advanced["recent_spend_90d"] + eps)
        / (advanced["historical_spend_rest"] + eps)
    ).clip(-5, 5)

    # Fill remaining NaNs with sensible defaults
    advanced["avg_days_between_txns"] = advanced["avg_days_between_txns"].fillna(
        OBSERVATION_PERIOD_DAYS
    )

    return advanced


def compute_target(
    df: pd.DataFrame,
    customer_ids: pd.Series,
    cut_off: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute the ground-truth CLV target for every customer.
    CLV = total spend in (cut_off, cut_off + TARGET_PERIOD_DAYS].
    Customers with zero activity in that window get CLV = 0.
    """
    future = df[
        (df["transaction_date"] > cut_off)
        & (df["transaction_date"] <= cut_off + pd.Timedelta(days=TARGET_PERIOD_DAYS))
    ]

    clv = (
        future.groupby("customer_id")["purchase_amount"]
        .sum()
        .reindex(customer_ids)
        .fillna(0)
        .reset_index()
        .rename(columns={"purchase_amount": "clv_90d"})
    )
    return clv


def _remove_outliers(df: pd.DataFrame, col: str, n_sigma: float = 5.0) -> pd.DataFrame:
    """
    Winsorise extreme values at n_sigma standard deviations from the mean.
    We use winsorising rather than removal to retain all customer rows.
    """
    mu, sigma = df[col].mean(), df[col].std()
    lower = mu - n_sigma * sigma
    upper = mu + n_sigma * sigma
    df[col] = df[col].clip(lower, upper)
    return df


def build_feature_matrix(
    path: str = "data/transactions.csv",
    cut_off: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: raw CSV → (features, target) DataFrames.

    Parameters
    ----------
    path     : path to raw transactions CSV
    cut_off  : ISO date string for the observation / target boundary.
               If None, automatically set to max_date − TARGET_PERIOD_DAYS.

    Returns
    -------
    features : DataFrame of engineered features (one row per customer)
    target   : DataFrame with customer_id and clv_90d columns
    """
    print("Loading transactions …")
    df = load_transactions(path)

    cut_off_ts = _set_cut_off(df, cut_off)
    print(f"Cut-off date: {cut_off_ts.date()}")

    print("Computing RFM features …")
    rfm = compute_rfm_features(df, cut_off_ts)

    print("Computing advanced behavioural features …")
    features = compute_advanced_features(df, rfm, cut_off_ts)

    print("Computing CLV target variable …")
    target = compute_target(df, features["customer_id"], cut_off_ts)

    # Merge target into features for convenience (modelling pipeline will split)
    full = features.merge(target, on="customer_id", how="left")
    full["clv_90d"] = full["clv_90d"].fillna(0)

    # ── outlier treatment on key numeric columns ───────────────────────────────
    for col in ["monetary", "clv_90d", "max_order_value", "avg_order_value"]:
        if col in full.columns:
            full = _remove_outliers(full, col)

    # ── drop non-feature columns before returning feature matrix ──────────────
    drop_cols = ["last_purchase_date", "first_purchase_date"]
    feature_df = full.drop(columns=drop_cols, errors="ignore")

    print(f"\nFeature matrix shape : {feature_df.shape}")
    print(f"Customers with CLV>0 : {(feature_df['clv_90d'] > 0).sum():,}")
    print(f"Average CLV (90d)    : ${feature_df['clv_90d'].mean():,.2f}")

    return feature_df, target


if __name__ == "__main__":
    features, target = build_feature_matrix()
    features.to_csv("data/features.csv", index=False)
    print("Features saved → data/features.csv")
