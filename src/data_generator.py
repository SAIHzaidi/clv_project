"""
data_generator.py
-----------------
Generates a realistic synthetic e-commerce transaction dataset
that mirrors patterns found in real-world retail data (e.g., UCI Online Retail).

Design decisions:
- Customer behaviour follows a Pareto distribution (80/20 rule):
  a small cohort drives the majority of revenue.
- Purchase amounts are log-normally distributed to capture the
  long tail of high-value orders.
- Churn is modelled probabilistically so the dataset contains
  both active and lapsed customers.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── simulation parameters ─────────────────────────────────────────────────────
N_CUSTOMERS    = 5_000
N_TRANSACTIONS = 60_000
START_DATE     = datetime(2022, 1, 1)
END_DATE       = datetime(2024, 3, 31)
SIMULATION_DAYS = (END_DATE - START_DATE).days


def _generate_customer_profiles(n: int) -> pd.DataFrame:
    """
    Create latent customer-level parameters that govern buying behaviour.
    Using a mixture model: ~20 % high-value, ~30 % medium, ~50 % low-value.
    """
    segments = np.random.choice(
        ["high", "medium", "low"],
        size=n,
        p=[0.20, 0.30, 0.50],
    )

    # Base purchase frequency (purchases per year) — drawn per segment
    freq_map   = {"high": (18, 6), "medium": (8, 3), "low": (2.5, 1.5)}
    spend_map  = {"high": (5.5, 0.5), "medium": (4.2, 0.5), "low": (3.5, 0.4)}

    annual_freq, avg_log_spend = [], []
    for seg in segments:
        mu_f, sd_f   = freq_map[seg]
        mu_s, sd_s   = spend_map[seg]
        annual_freq.append(max(1, np.random.normal(mu_f, sd_f)))
        avg_log_spend.append(np.random.normal(mu_s, sd_s))

    # Acquisition date: spread uniformly across the observation window
    acq_offset = np.random.randint(0, SIMULATION_DAYS - 30, size=n)
    acq_dates  = [START_DATE + timedelta(days=int(d)) for d in acq_offset]

    # Churn probability (some customers lapse before END_DATE)
    churn_p = {"high": 0.15, "medium": 0.30, "low": 0.55}
    churned = [np.random.rand() < churn_p[s] for s in segments]

    return pd.DataFrame({
        "customer_id"     : [f"CUST_{i:05d}" for i in range(n)],
        "segment"         : segments,
        "annual_freq"     : annual_freq,
        "avg_log_spend"   : avg_log_spend,
        "acquisition_date": acq_dates,
        "churned"         : churned,
    })


def _generate_transactions(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate individual transactions for every customer based on their
    latent profile.  Returns a long-format transactions table.
    """
    records = []

    for _, cust in profiles.iterrows():
        acq        = cust["acquisition_date"]
        days_active = (END_DATE - acq).days

        if days_active <= 0:
            continue

        # If churned, customer only buys during the first fraction of their
        # potential active window
        if cust["churned"]:
            days_active = max(1, int(days_active * np.random.uniform(0.05, 0.60)))

        # Expected number of purchases in the active window
        expected_n = max(1, int(cust["annual_freq"] * days_active / 365))
        n_purchases = np.random.poisson(expected_n)

        if n_purchases == 0:
            continue

        # Purchase timestamps: sorted uniform draws within the active window
        offsets = sorted(np.random.randint(0, days_active, size=n_purchases))

        for offset in offsets:
            txn_date = acq + timedelta(days=int(offset))
            if txn_date > END_DATE:
                break

            # Amount: log-normal, with occasional bulk / discount orders
            base_amount = np.exp(np.random.normal(cust["avg_log_spend"], 0.6))

            # ~5 % chance of a spike purchase (e.g. holiday, gifting)
            if np.random.rand() < 0.05:
                base_amount *= np.random.uniform(2, 5)

            # ~3 % chance of a very small purchase (add-on, small accessory)
            if np.random.rand() < 0.03:
                base_amount *= np.random.uniform(0.1, 0.3)

            records.append({
                "customer_id"     : cust["customer_id"],
                "transaction_date": txn_date,
                "purchase_amount" : round(base_amount, 2),
                "true_segment"    : cust["segment"],     # kept for evaluation
            })

    return pd.DataFrame(records)


def generate_dataset(save_path: str = "data/transactions.csv") -> pd.DataFrame:
    """
    End-to-end dataset generation pipeline.
    Saves raw transactions to CSV and returns the DataFrame.
    """
    print("Generating customer profiles …")
    profiles = _generate_customer_profiles(N_CUSTOMERS)

    print("Simulating transactions …")
    transactions = _generate_transactions(profiles)

    # Ensure chronological ordering
    transactions.sort_values(["customer_id", "transaction_date"], inplace=True)
    transactions.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    transactions.to_csv(save_path, index=False)

    print(f"Dataset saved → {save_path}")
    print(f"  Customers   : {transactions['customer_id'].nunique():,}")
    print(f"  Transactions: {len(transactions):,}")
    print(f"  Date range  : {transactions['transaction_date'].min().date()} "
          f"→ {transactions['transaction_date'].max().date()}")

    return transactions


if __name__ == "__main__":
    generate_dataset()
