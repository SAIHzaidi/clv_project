"""
main.py  –  FastAPI REST API for CLV Prediction
------------------------------------------------
Endpoints
---------
GET  /              → health check
GET  /feature-names → list of expected feature names
POST /predict       → predict CLV for one customer
POST /predict-batch → predict CLV for multiple customers (CSV upload)
GET  /feature-importance → top-N feature importance values

Run locally
-----------
    uvicorn api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── bootstrap path so imports work from project root ─────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictor import CLVPredictor

# ── global model instance (loaded once at startup) ───────────────────────────
predictor: Optional[CLVPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artefacts on startup, release on shutdown."""
    global predictor
    try:
        predictor = CLVPredictor()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("  Run src/train.py first to generate model artefacts.")
    yield
    predictor = None


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Lifetime Value (CLV) Prediction API",
    description=(
        "Predicts 90-day Customer Lifetime Value from behavioural "
        "RFM features and returns a customer segment with actionable "
        "business recommendations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── schemas ───────────────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """Input schema — all features are optional; missing ones default to 0."""
    frequency             : float = Field(0, description="Number of purchases in observation window")
    monetary              : float = Field(0, description="Total spend in observation window ($)")
    recency               : float = Field(365, description="Days since last purchase")
    customer_tenure       : float = Field(0, description="Days since first purchase")
    avg_order_value       : float = Field(0, description="Mean spend per transaction ($)")
    std_order_value       : float = Field(0, description="Standard deviation of order values")
    max_order_value       : float = Field(0, description="Largest single purchase ($)")
    min_order_value       : float = Field(0, description="Smallest single purchase ($)")
    cv_order_value        : float = Field(0, description="Coefficient of variation of order values")
    avg_days_between_txns : float = Field(365, description="Average inter-purchase gap (days)")
    recent_spend_90d      : float = Field(0, description="Spend in last 90 days of obs window ($)")
    historical_spend_rest : float = Field(0, description="Spend outside last 90 days of obs window ($)")
    spend_trend           : float = Field(0, description="Log ratio recent / historical spend")
    purchase_rate         : float = Field(0, description="Purchases per active day")
    months_with_purchase  : float = Field(0, description="Number of distinct months with a purchase")

    class Config:
        json_schema_extra = {
            "example": {
                "frequency"            : 12,
                "monetary"             : 850.00,
                "recency"              : 14,
                "customer_tenure"      : 365,
                "avg_order_value"      : 70.83,
                "std_order_value"      : 22.10,
                "max_order_value"      : 145.00,
                "min_order_value"      : 15.00,
                "cv_order_value"       : 0.31,
                "avg_days_between_txns": 28.5,
                "recent_spend_90d"     : 210.00,
                "historical_spend_rest": 640.00,
                "spend_trend"          : -0.30,
                "purchase_rate"        : 0.033,
                "months_with_purchase" : 10,
            }
        }


class PredictionResponse(BaseModel):
    predicted_clv  : float
    segment        : str
    segment_label  : str
    segment_emoji  : str
    recommendations: List[str]
    confidence_note: str


class HealthResponse(BaseModel):
    status     : str
    model_ready: bool
    version    : str


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="ok",
        model_ready=predictor is not None,
        version="1.0.0",
    )


@app.get("/feature-names", response_model=List[str], tags=["Model Info"])
def get_feature_names():
    """Returns the ordered list of feature names expected by the model."""
    _require_model()
    return predictor.feature_names


@app.get("/feature-importance", response_model=List[Dict[str, Any]], tags=["Model Info"])
def get_feature_importance(top_n: int = 15):
    """Returns top-N feature importance scores from the best model."""
    _require_model()
    return predictor.feature_importance(top_n=top_n)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(customer: CustomerFeatures):
    """
    Predict 90-day CLV for a single customer.

    Returns predicted CLV, segment (HIGH/MEDIUM/LOW), and a list of
    actionable business recommendations for that segment.
    """
    _require_model()
    features = customer.model_dump()
    result   = predictor.predict(features)
    return PredictionResponse(**result)


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """
    Predict CLV for multiple customers from an uploaded CSV file.

    The CSV must contain at least some of the feature columns listed in
    /feature-names.  Missing columns are imputed with 0.
    Returns a JSON list of predictions.
    """
    _require_model()
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    try:
        df = pd.read_csv(pd.io.common.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    result_df = predictor.predict_batch(df)
    return result_df[["predicted_clv", "segment", "segment_label"]].to_dict(orient="records")


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_model():
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run src/train.py first to generate artefacts.",
        )


# ── dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
