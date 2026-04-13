# Customer Lifetime Value (CLV) Prediction System

> **End-to-end ML system that predicts 90-day Customer Lifetime Value from transactional data, segments customers into actionable tiers, and serves predictions via a REST API and interactive dashboard.**

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Approach](#approach)
3. [Project Structure](#project-structure)
4. [Feature Engineering](#feature-engineering)
5. [Model Comparison](#model-comparison)
6. [Results](#results)
7. [Business Impact](#business-impact)
8. [How to Run Locally](#how-to-run-locally)
9. [API Reference](#api-reference)
10. [Deployment](#deployment)

---

## Problem Statement

Customer Lifetime Value (CLV) is the single most important metric for sustainable growth in e-commerce and subscription businesses. Companies that can accurately predict which customers will generate the most revenue over the next quarter can:

- **Allocate marketing budgets** with precision — spend more on retaining high-CLV customers, less on low-CLV ones.
- **Personalise experiences** at scale — segment-specific offers, messaging, and support tiers.
- **Improve unit economics** — reduce CAC:LTV ratio by acquiring customers that look like your best ones.

**The business question:** *Given a customer's historical purchase behaviour, how much revenue will they generate over the next 90 days?*

Without a CLV model, teams rely on recency/frequency rules of thumb that miss non-obvious signals (e.g. a customer with erratic but high-value purchases vs. a frequent low-spender).

---

## Approach

### Methodology

We frame CLV prediction as a **regression problem** on a fixed future horizon (90 days). The pipeline:

```
Raw Transactions → Feature Engineering → Model Training → Segmentation → API / App
```

**Why 90 days?**
90 days balances predictive accuracy (short enough to have signal) with business relevance (long enough to take action). The horizon is configurable in `src/feature_engineering.py`.

**Target variable:** `clv_90d` = total spend in the 90 days *after* the observation cut-off date. Customers with no purchases in this window receive CLV = 0, correctly capturing churn.

**Training strategy:** We apply `log1p` transformation to the target during training. CLV follows a power-law distribution (many zeros, long tail of high spenders). Log-transforming stabilises gradients and prevents the model from over-optimising for rare outliers at the expense of the median customer.

---

## Project Structure

```
clv_project/
├── data/
│   ├── transactions.csv          # Raw synthetic transaction data
│   └── features.csv              # Engineered feature matrix
│
├── src/
│   ├── data_generator.py         # Synthetic data generation (Pareto-distributed)
│   ├── feature_engineering.py    # RFM + advanced feature pipeline
│   ├── train.py                  # Full training (sklearn + LightGBM)
│   ├── train_fast.py             # Lightweight training (numpy only)
│   ├── segmentation.py           # CLV-based customer segmentation
│   └── predictor.py              # Inference wrapper (used by API + app)
│
├── api/
│   └── main.py                   # FastAPI REST API
│
├── app/
│   └── streamlit_app.py          # Streamlit dashboard
│
├── models/
│   ├── best_model.pkl            # Best serialised model
│   ├── feature_names.pkl         # Ordered feature list
│   ├── feature_importance.csv    # Feature importance scores
│   ├── training_report.json      # Metrics & CV results
│   ├── test_predictions.csv      # Test-set predictions
│   └── segmented_customers.csv   # Customers with segment labels
│
├── run_pipeline.py               # Master pipeline runner
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Feature Engineering

Features are computed over a configurable **observation window** (default: 365 days) ending at a cut-off date.

### RFM Features

| Feature | Description |
|---------|-------------|
| `recency` | Days since most recent purchase |
| `frequency` | Number of distinct purchase events |
| `monetary` | Total spend in observation window |

### Advanced Behavioural Features

| Feature | Description | Business Signal |
|---------|-------------|-----------------|
| `avg_order_value` | Mean spend per transaction | Typical basket size |
| `std_order_value` | Standard deviation of order values | Purchase consistency |
| `max_order_value` | Largest single purchase | Whale potential |
| `cv_order_value` | Coefficient of variation | Erratic vs. predictable buyer |
| `avg_days_between_txns` | Mean inter-purchase gap | Purchase cadence |
| `customer_tenure` | Days since first purchase | Loyalty / lifecycle stage |
| `purchase_rate` | Purchases per active day | Normalised engagement |
| `months_with_purchase` | Calendar months with ≥1 purchase | Seasonal breadth |
| `recent_spend_90d` | Spend in last 90 days of obs. window | Recent momentum |
| `historical_spend_rest` | Spend in earlier part of obs. window | Baseline behaviour |
| `spend_trend` | `log(recent / historical)` | Acceleration / deceleration |

---

## Model Comparison

Three models were trained and evaluated on a held-out test set (20% of customers):

| Model | CV RMSE (log scale) | Test RMSE ($) | Test MAE ($) |
|-------|---------------------|---------------|--------------|
| Ridge Regression | 2.316 | 22,987 | 1,929 |
| Random Forest | 2.117 | 836 | 336 |
| **Gradient Boosting** | **1.962** | **747** | **305** |

**Winner: Gradient Boosting** — 96.7% lower RMSE than the Ridge baseline.

### Why Gradient Boosting wins

- Handles non-linear interactions between features (e.g. high frequency + high recency = very different CLV than either alone).
- Robust to the class imbalance between zero-CLV and positive-CLV customers.
- Subsample-based training provides implicit regularisation, preventing overfitting on the outlier tail.

### Top Features (by importance)

1. `recency` — most powerful single predictor; recent buyers buy again
2. `months_with_purchase` — breadth of engagement signals loyalty
3. `cv_order_value` — purchase consistency predicts future behaviour
4. `recent_spend_90d` — momentum is a strong forward indicator
5. `spend_trend` — acceleration predicts above-baseline future spend

---

## Results

### Segment Distribution (Test Set, n=670)

| Segment | Customers | Mean CLV | Revenue Share |
|---------|-----------|----------|---------------|
| 💎 High Value | 134 (20%) | $930 | **90.6%** |
| ⭐ Medium Value | 268 (40%) | $47 | 9.1% |
| 🌱 Low Value | 268 (40%) | $1.41 | 0.3% |

The model successfully identifies the **Pareto effect**: the top 20% of customers account for over 90% of predicted revenue — a pattern consistent with real-world e-commerce data.

---

## Business Impact

### Quantified value of CLV segmentation

Assume a company with 10,000 customers and a marketing budget of $50,000/quarter:

**Without CLV model (spray-and-pray):**
- Equal spend: $5 per customer
- Expected return: moderate, unfocused

**With CLV model:**
- Invest $30 on High-Value (2,000 customers) → Defend $9M+ predicted revenue
- Invest $10 on Medium-Value (3,000 customers) → Convert 10–15% to High tier
- Invest $2 on Low-Value (5,000 customers) → Automated nurture only

**Outcome:** Higher ROI on retention spend, better allocation of CS resources, personalised onboarding for high-CLV new customers.

### Segment Playbooks

**💎 High Value** — *Retain at all costs*
- VIP programme, dedicated account management, early product access
- Proactive service recovery before issues escalate

**⭐ Medium Value** — *Upgrade journey*
- Loyalty programme enrolment, cross-sell bundles
- Personalised email sequences after 30 days of inactivity

**🌱 Low Value** — *Efficient nurture*
- Automated win-back flows (60-day trigger)
- Low-cost channels: push notifications, SMS
- Churn risk scoring for proactive intervention

---

## How to Run Locally

### Prerequisites

```bash
Python 3.10+
```

### 1. Clone and install

```bash
git clone https://github.com/yourname/clv-prediction.git
cd clv-prediction
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_pipeline.py
```

This will:
- Generate ~34,000 synthetic transactions for 4,200 customers
- Engineer 15 behavioural features per customer
- Train Ridge, Random Forest, and Gradient Boosting models
- Evaluate on a held-out test set and select the best model
- Segment customers and generate business recommendations
- Save all artefacts to `models/`

Expected runtime: **3–5 minutes** (sklearn + LightGBM) or **2–3 minutes** (fast numpy version).

### 3. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

### 4. Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

Dashboard available at: `http://localhost:8501`

---

## API Reference

### `POST /predict`

Predict CLV for a single customer.

**Request body:**
```json
{
  "frequency": 12,
  "monetary": 850.00,
  "recency": 14,
  "customer_tenure": 365,
  "avg_order_value": 70.83,
  "avg_days_between_txns": 28.5,
  "recent_spend_90d": 210.00,
  "historical_spend_rest": 640.00,
  "months_with_purchase": 10
}
```

**Response:**
```json
{
  "predicted_clv": 312.45,
  "segment": "HIGH",
  "segment_label": "High Value",
  "segment_emoji": "💎",
  "recommendations": [
    "Enrol in a VIP / loyalty programme with exclusive perks.",
    "..."
  ],
  "confidence_note": "Prediction based on historical transaction patterns..."
}
```

### `POST /predict-batch`

Upload a CSV file; returns predictions for all customers.

### `GET /feature-importance`

Returns top-N feature importance scores.

---

## Deployment

### Docker (local)

```bash
# Build and run both API + Streamlit
docker-compose up --build

# API:       http://localhost:8000
# Streamlit: http://localhost:8501
```

### Deploy to Render

1. Push repository to GitHub.
2. Create a new **Web Service** on [render.com](https://render.com).
3. Set **Build Command:** `pip install -r requirements.txt && python run_pipeline.py`
4. Set **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. For Streamlit: create a separate service with start command:
   `streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

### Deploy to Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

railway login
railway init
railway up
```

Set environment variables via the Railway dashboard.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data processing | pandas, numpy |
| ML models | scikit-learn, LightGBM |
| Model serialisation | joblib / pickle |
| REST API | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Containerisation | Docker + docker-compose |
| Deployment | Render / Railway |

---

## Author

I built this as a production-quality portfolio project demonstrating end-to-end ML engineering: data generation, feature design, model selection, REST API development, and interactive visualisation.

let me know if theres any issues with use. 

SAIHZaidi

