"""
streamlit_app.py  –  CLV Prediction Dashboard
----------------------------------------------
A production-quality Streamlit application for exploring CLV predictions.

Tabs
----
1. Single Prediction  — manual feature entry → instant CLV + segment
2. Batch Prediction   — CSV upload → bulk predictions + export
3. Model Insights     — feature importance, training report
4. Segment Analysis   — CLV distribution, segment breakdown
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── path bootstrap ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictor import CLVPredictor
from src.segmentation import SEGMENT_RECOMMENDATIONS

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLV Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 5px 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.8; margin-top: 4px; }
    .segment-high   { background: linear-gradient(135deg,#1a6e3c,#27ae60); }
    .segment-medium { background: linear-gradient(135deg,#7d4a00,#e67e22); }
    .segment-low    { background: linear-gradient(135deg,#4a1942,#8e44ad); }
    .rec-item { background:#f8f9fa; border-left:4px solid #2d6a9f;
                padding:8px 12px; margin:4px 0; border-radius:0 6px 6px 0;
                font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)


# ── load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_predictor() -> CLVPredictor | None:
    try:
        return CLVPredictor()
    except Exception as e:
        st.error(f"Could not load model: {e}. Run `python run_pipeline.py` first.")
        return None


predictor = load_predictor()

SEGMENT_COLORS = {"HIGH": "#27ae60", "MEDIUM": "#e67e22", "LOW": "#8e44ad"}
SEGMENT_CSS    = {"HIGH": "segment-high", "MEDIUM": "segment-medium", "LOW": "segment-low"}


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/money-bag.png", width=64)
    st.title("CLV Prediction\nSystem")
    st.caption("v1.0.0 · 90-day horizon")
    st.divider()
    st.markdown("**Model status**")
    if predictor:
        st.success("✓ Model loaded")
    else:
        st.error("✗ Model not found")
    st.divider()
    st.markdown("**About**")
    st.caption(
        "Predicts 90-day Customer Lifetime Value using RFM + behavioural "
        "features. Powered by LightGBM with cross-validated hyperparameters."
    )


# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Prediction",
    "📦 Batch Prediction",
    "🔬 Model Insights",
    "📊 Segment Analysis",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Single Customer CLV Prediction")
    st.caption("Enter customer features to receive an instant CLV estimate and segment.")

    if predictor is None:
        st.warning("Model not loaded. Please run the training pipeline first.")
    else:
        col_l, col_r = st.columns([1, 1], gap="large")

        with col_l:
            st.subheader("📋 Core RFM Features")
            frequency   = st.slider("Purchase Frequency (last 12 months)", 1, 100, 12)
            monetary    = st.number_input("Total Spend ($)", 0.0, 50000.0, 850.0, step=10.0)
            recency     = st.slider("Recency (days since last purchase)", 1, 365, 14)
            tenure      = st.slider("Customer Tenure (days)", 1, 1825, 365)

            st.subheader("📈 Behavioural Features")
            avg_order   = st.number_input("Avg Order Value ($)", 0.0, 5000.0, 70.83, step=5.0)
            std_order   = st.number_input("Std Dev Order Value ($)", 0.0, 2000.0, 22.10, step=1.0)
            max_order   = st.number_input("Max Order Value ($)", 0.0, 10000.0, 145.0, step=5.0)
            avg_gap     = st.slider("Avg Days Between Purchases", 1, 365, 30)
            purchase_rt = st.number_input("Purchase Rate (per day)", 0.0, 1.0, 0.033, format="%.4f")
            months_p    = st.slider("Months with a Purchase", 1, 24, 10)

        with col_r:
            st.subheader("💹 Trend Features")
            recent_90  = st.number_input("Spend — Last 90 Days ($)", 0.0, 10000.0, 210.0, step=10.0)
            hist_rest  = st.number_input("Spend — Prior Period ($)", 0.0, 50000.0, 640.0, step=10.0)
            spend_trend = np.log((recent_90 + 1e-3) / (hist_rest + 1e-3))
            st.info(f"ℹ️ Computed spend trend: **{spend_trend:.3f}** (log ratio)")

            st.divider()
            if st.button("🔮 Predict CLV", use_container_width=True, type="primary"):
                features = {
                    "frequency"            : frequency,
                    "monetary"             : monetary,
                    "recency"              : recency,
                    "customer_tenure"      : tenure,
                    "avg_order_value"      : avg_order,
                    "std_order_value"      : std_order,
                    "max_order_value"      : max_order,
                    "min_order_value"      : max(0, avg_order - std_order),
                    "cv_order_value"       : std_order / max(avg_order, 1),
                    "avg_days_between_txns": avg_gap,
                    "recent_spend_90d"     : recent_90,
                    "historical_spend_rest": hist_rest,
                    "spend_trend"          : spend_trend,
                    "purchase_rate"        : purchase_rt,
                    "months_with_purchase" : months_p,
                }
                result = predictor.predict(features)
                seg    = result["segment"]
                css    = SEGMENT_CSS[seg]

                st.markdown(f"""
                <div class="metric-card {css}">
                    <div class="metric-value">${result['predicted_clv']:,.2f}</div>
                    <div class="metric-label">Predicted 90-Day CLV</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-card {css}">
                    <div class="metric-value">{result['segment_emoji']} {result['segment_label']}</div>
                    <div class="metric-label">Customer Segment</div>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("💡 Recommended Actions")
                rec_data = SEGMENT_RECOMMENDATIONS[seg]
                st.caption(rec_data["description"])
                for action in result["recommendations"]:
                    st.markdown(f'<div class="rec-item">→ {action}</div>', unsafe_allow_html=True)

                st.caption(f"ℹ️ {result['confidence_note']}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Batch Customer CLV Prediction")
    st.caption("Upload a CSV containing customer features for bulk scoring.")

    if predictor is None:
        st.warning("Model not loaded.")
    else:
        # Show sample template
        with st.expander("📄 Download CSV Template"):
            template_cols = predictor.feature_names
            sample = pd.DataFrame([{c: 0.0 for c in template_cols}])
            st.dataframe(sample)
            csv_bytes = sample.to_csv(index=False).encode()
            st.download_button("⬇️ Download Template", csv_bytes, "clv_template.csv", "text/csv")

        uploaded = st.file_uploader("Upload customer data (CSV)", type=["csv"])

        if uploaded:
            raw = pd.read_csv(uploaded)
            st.write(f"**{len(raw):,} customers** loaded")
            st.dataframe(raw.head(), use_container_width=True)

            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Scoring …"):
                    results = predictor.predict_batch(raw)

                st.success(f"✓ {len(results):,} predictions completed")

                # KPI summary
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Predicted Revenue", f"${results['predicted_clv'].sum():,.0f}")
                k2.metric("Avg CLV",                 f"${results['predicted_clv'].mean():,.2f}")
                k3.metric("High-Value Customers",    (results["segment"] == "HIGH").sum())
                k4.metric("Low-Value Customers",     (results["segment"] == "LOW").sum())

                # CLV distribution
                fig = px.histogram(
                    results, x="predicted_clv", color="segment",
                    color_discrete_map=SEGMENT_COLORS,
                    nbins=50, title="Predicted CLV Distribution by Segment",
                    labels={"predicted_clv": "Predicted 90-Day CLV ($)"},
                )
                fig.update_layout(bargap=0.05)
                st.plotly_chart(fig, use_container_width=True)

                # Segment pie
                seg_counts = results["segment"].value_counts().reset_index()
                seg_counts.columns = ["segment", "count"]
                fig2 = px.pie(
                    seg_counts, values="count", names="segment",
                    color="segment", color_discrete_map=SEGMENT_COLORS,
                    title="Customer Segment Distribution",
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Full results table
                st.subheader("Full Results")
                display_df = results[["predicted_clv", "segment", "segment_label"]].copy()
                if "customer_id" in results.columns:
                    display_df.insert(0, "customer_id", results["customer_id"])
                st.dataframe(display_df, use_container_width=True)

                # Download
                out_csv = display_df.to_csv(index=False).encode()
                st.download_button("⬇️ Download Results", out_csv, "clv_predictions.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Model Insights")

    report_path = Path("models/training_report.json")
    fi_path     = Path("models/feature_importance.csv")

    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

        st.subheader("🏆 Training Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Best Model",    report.get("best_model", "—"))
        m2.metric("Train Samples", f"{report.get('n_train', 0):,}")
        m3.metric("Test Samples",  f"{report.get('n_test', 0):,}")

        # Model comparison table
        st.subheader("📊 Model Comparison (Test Set)")
        metrics = report.get("test_metrics", {})
        if metrics:
            comp_df = pd.DataFrame(metrics).T.reset_index()
            comp_df.columns = ["Model", "RMSE ($)", "MAE ($)"]
            comp_df = comp_df.sort_values("RMSE ($)")
            st.dataframe(
                comp_df.style.highlight_min(subset=["RMSE ($)", "MAE ($)"], color="#d4edda"),
                use_container_width=True,
            )

            fig = go.Figure()
            for metric, color in [("RMSE ($)", "#e74c3c"), ("MAE ($)", "#3498db")]:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=comp_df["Model"],
                    y=comp_df[metric],
                    marker_color=color,
                ))
            fig.update_layout(
                barmode="group",
                title="RMSE & MAE by Model",
                yaxis_title="Error ($)",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the training pipeline to see model metrics.")

    if fi_path.exists() and predictor:
        st.subheader("🎯 Feature Importance")
        fi_df = pd.read_csv(fi_path).head(15)
        fig3 = px.bar(
            fi_df.sort_values("importance"),
            x="importance", y="feature",
            orientation="h",
            title="Top 15 Features (Best Model)",
            color="importance",
            color_continuous_scale="Blues",
        )
        fig3.update_layout(showlegend=False, yaxis_title="")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Feature importance will appear after training.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Segment Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Customer Segment Analysis")

    pred_path = Path("models/test_predictions.csv")
    if pred_path.exists():
        test_df = pd.read_csv(pred_path)
        if "y_pred_best" in test_df.columns:
            test_df["predicted_clv"] = test_df["y_pred_best"]

        # Assign segments using the same bins as predictor
        test_df["segment"] = pd.cut(
            test_df["predicted_clv"],
            bins=[-np.inf, 80, 300, np.inf],
            labels=["LOW", "MEDIUM", "HIGH"],
        ).astype(str)

        # CLV box plot by segment
        fig4 = px.box(
            test_df, x="segment", y="predicted_clv",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title="CLV Distribution by Segment",
            labels={"predicted_clv": "Predicted 90-Day CLV ($)", "segment": "Segment"},
            category_orders={"segment": ["LOW", "MEDIUM", "HIGH"]},
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Revenue contribution
        rev = (
            test_df.groupby("segment")["predicted_clv"]
            .agg(total_rev="sum", n_cust="count")
            .reset_index()
        )
        rev["avg_clv"]   = (rev["total_rev"] / rev["n_cust"]).round(2)
        rev["rev_share"] = (rev["total_rev"] / rev["total_rev"].sum() * 100).round(1)

        col_a, col_b = st.columns(2)
        with col_a:
            fig5 = px.pie(
                rev, values="total_rev", names="segment",
                color="segment", color_discrete_map=SEGMENT_COLORS,
                title="Revenue Share by Segment",
            )
            st.plotly_chart(fig5, use_container_width=True)
        with col_b:
            st.subheader("Segment KPIs")
            st.dataframe(
                rev.rename(columns={
                    "segment"  : "Segment",
                    "total_rev": "Total Revenue ($)",
                    "n_cust"   : "Customers",
                    "avg_clv"  : "Avg CLV ($)",
                    "rev_share": "Revenue Share (%)",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # Actual vs Predicted scatter
        if "y_true" in test_df.columns:
            fig6 = px.scatter(
                test_df.sample(min(500, len(test_df))),
                x="y_true", y="predicted_clv",
                color="segment", color_discrete_map=SEGMENT_COLORS,
                title="Actual vs Predicted CLV (test sample)",
                labels={"y_true": "Actual CLV ($)", "predicted_clv": "Predicted CLV ($)"},
                opacity=0.6,
            )
            # Perfect prediction line
            max_val = max(test_df["y_true"].max(), test_df["predicted_clv"].max())
            fig6.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="black", dash="dash"),
            ))
            st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Run the training pipeline to see segment analysis.")

    # Business recommendations card
    st.subheader("💼 Business Recommendations by Segment")
    for seg, data in SEGMENT_RECOMMENDATIONS.items():
        with st.expander(f"{data['emoji']} {data['label']} Customers"):
            st.write(data["description"])
            st.markdown(f"**Retention Risk:** {data['retention_risk']}")
            st.markdown(f"**Expected ROI:** {data['expected_roi']}")
            st.markdown("**Actions:**")
            for action in data["recommended_actions"]:
                st.markdown(f"- {action}")
