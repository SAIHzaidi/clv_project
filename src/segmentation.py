"""
segmentation.py
---------------
Assigns customers to CLV-based value tiers and generates
actionable business recommendations for each segment.

Segmentation approach
---------------------
We use quantile-based thresholds rather than arbitrary dollar amounts
so the segmentation adapts gracefully to different industries /
revenue scales.

  • HIGH   – top 20 % by predicted CLV
  • MEDIUM – next 40 % (20th–60th percentile)
  • LOW    – bottom 40 %

Business context
----------------
Typical CLV segmentation use-cases:
  - HIGH   → retention investment, VIP programmes, proactive support
  - MEDIUM → upgrade campaigns, loyalty programme onboarding
  - LOW    → low-cost nurture flows, re-engagement, churn prediction
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── segment thresholds (quantile-based, set at scoring time) ─────────────────
HIGH_QUANTILE   = 0.80   # top 20 %
MEDIUM_QUANTILE = 0.40   # 40th–80th percentile


SEGMENT_RECOMMENDATIONS = {
    "HIGH": {
        "label"           : "High Value",
        "emoji"           : "💎",
        "description"     : (
            "These customers represent your most valuable cohort. "
            "They purchase frequently, spend significantly, and are "
            "likely brand advocates."
        ),
        "retention_risk"  : "Low to Medium",
        "recommended_actions": [
            "Enrol in a VIP / loyalty programme with exclusive perks.",
            "Provide dedicated customer success or account management.",
            "Offer early access to new products / beta features.",
            "Personalise communications at the individual level.",
            "Proactively resolve any service issues before escalation.",
            "Survey for Net Promoter Score and referral programme.",
        ],
        "expected_roi"    : "High — small investment yields outsized retention value.",
    },
    "MEDIUM": {
        "label"           : "Medium Value",
        "emoji"           : "⭐",
        "description"     : (
            "Solid, reliable customers with growth potential. "
            "Well-designed upgrade journeys can convert a meaningful "
            "fraction into High-Value customers."
        ),
        "retention_risk"  : "Medium",
        "recommended_actions": [
            "Introduce loyalty programme to increase purchase frequency.",
            "Cross-sell complementary products based on purchase history.",
            "Send personalised re-engagement emails after 30 days of inactivity.",
            "Offer bundle discounts to increase average order value.",
            "A/B test incentive structures (% off vs. free shipping).",
        ],
        "expected_roi"    : "Medium — scalable automation with targeted personalisation.",
    },
    "LOW": {
        "label"           : "Low Value",
        "emoji"           : "🌱",
        "description"     : (
            "Customers with limited engagement or early in their lifecycle. "
            "Cost-efficient nurture flows are appropriate; "
            "identify the subset with churn risk for re-activation."
        ),
        "retention_risk"  : "High",
        "recommended_actions": [
            "Trigger automated win-back email sequences after 60 days inactive.",
            "Offer a one-time discount to encourage a second purchase.",
            "Use low-cost channels (push notifications, SMS) for outreach.",
            "Analyse purchase patterns for early churn signals.",
            "Avoid over-investing — focus resources on segments above.",
            "Run periodic surveys to understand barriers to engagement.",
        ],
        "expected_roi"    : "Low individually — optimise for scale and automation.",
    },
}


def assign_segments(
    predicted_clv: pd.Series,
    high_q: float = HIGH_QUANTILE,
    medium_q: float = MEDIUM_QUANTILE,
) -> pd.Series:
    """
    Assign a segment label to each customer based on predicted CLV quantile.

    Parameters
    ----------
    predicted_clv : Series of predicted CLV values (one per customer)
    high_q        : quantile threshold above which customers are HIGH
    medium_q      : quantile threshold above which customers are MEDIUM

    Returns
    -------
    Series of segment labels: 'HIGH', 'MEDIUM', or 'LOW'
    """
    high_threshold   = predicted_clv.quantile(high_q)
    medium_threshold = predicted_clv.quantile(medium_q)

    def _label(clv: float) -> str:
        if clv >= high_threshold:
            return "HIGH"
        elif clv >= medium_threshold:
            return "MEDIUM"
        else:
            return "LOW"

    return predicted_clv.apply(_label)


def segment_summary(
    df: pd.DataFrame,
    clv_col: str = "predicted_clv",
    segment_col: str = "segment",
) -> pd.DataFrame:
    """
    Compute aggregate statistics per segment for reporting.
    """
    summary = (
        df.groupby(segment_col)[clv_col]
        .agg(
            n_customers="count",
            mean_clv="mean",
            median_clv="median",
            total_clv="sum",
            min_clv="min",
            max_clv="max",
        )
        .reset_index()
    )
    total = summary["total_clv"].sum()
    summary["revenue_share_pct"] = (summary["total_clv"] / total * 100).round(1)
    return summary


def add_segment_metadata(df: pd.DataFrame, segment_col: str = "segment") -> pd.DataFrame:
    """Attach human-readable metadata columns to the customer DataFrame."""
    df = df.copy()
    df["segment_label"]       = df[segment_col].map(
        lambda s: SEGMENT_RECOMMENDATIONS[s]["label"]
    )
    df["segment_emoji"]       = df[segment_col].map(
        lambda s: SEGMENT_RECOMMENDATIONS[s]["emoji"]
    )
    return df


def print_recommendations(segment: str) -> None:
    """Pretty-print business recommendations for a given segment."""
    rec = SEGMENT_RECOMMENDATIONS.get(segment.upper())
    if rec is None:
        print(f"Unknown segment: {segment}")
        return

    print(f"\n{'='*60}")
    print(f"{rec['emoji']}  {rec['label']} Customer Segment")
    print(f"{'='*60}")
    print(f"\n{rec['description']}")
    print(f"\nRetention Risk : {rec['retention_risk']}")
    print(f"Expected ROI   : {rec['expected_roi']}")
    print("\nRecommended Actions:")
    for i, action in enumerate(rec["recommended_actions"], 1):
        print(f"  {i}. {action}")


def run_segmentation(
    predictions_path: Optional[str] = None,
    output_path: str = "models/segmented_customers.csv",
) -> pd.DataFrame:
    """
    Load model predictions, assign segments, save enriched output.
    Designed to run after train.py has produced test_predictions.csv.
    """
    path = predictions_path or "models/test_predictions.csv"
    df   = pd.read_csv(path)

    if "y_pred_best" in df.columns:
        df["predicted_clv"] = df["y_pred_best"]
    elif "predicted_clv" not in df.columns:
        raise ValueError("No prediction column found. Run train.py first.")

    df["segment"]       = assign_segments(df["predicted_clv"])
    df = add_segment_metadata(df)

    summary = segment_summary(df)
    print("\n── Segment Summary ─────────────────────────────────────────────")
    print(summary.to_string(index=False))

    for seg in ["HIGH", "MEDIUM", "LOW"]:
        print_recommendations(seg)

    df.to_csv(output_path, index=False)
    print(f"\n✓ Segmented data saved → {output_path}")
    return df


if __name__ == "__main__":
    run_segmentation()
