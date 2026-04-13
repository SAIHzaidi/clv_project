"""
run_pipeline.py
---------------
Master script that runs the complete CLV prediction pipeline end-to-end:

  1. Generate synthetic data      → data/transactions.csv
  2. Engineer features            → data/features.csv
  3. Train & evaluate models      → models/
  4. Run customer segmentation    → models/segmented_customers.csv

Run from the project root:
    python run_pipeline.py
"""

import sys
import time
from pathlib import Path

# ── path bootstrap ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))


def banner(title: str) -> None:
    width = 60
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def main():
    total_start = time.time()

    # ── Step 1: Generate data ─────────────────────────────────────────────────
    banner("STEP 1 / 4 — DATA GENERATION")
    from src.data_generator import generate_dataset
    generate_dataset(save_path="data/transactions.csv")

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    banner("STEP 2 / 4 — FEATURE ENGINEERING")
    from src.feature_engineering import build_feature_matrix
    features, _ = build_feature_matrix(
        path="data/transactions.csv",
    )
    features.to_csv("data/features.csv", index=False)

    # ── Step 3: Model training ────────────────────────────────────────────────
    banner("STEP 3 / 4 — MODEL TRAINING & EVALUATION")
    from src.train import train
    train()

    # ── Step 4: Customer segmentation ─────────────────────────────────────────
    banner("STEP 4 / 4 — CUSTOMER SEGMENTATION")
    from src.segmentation import run_segmentation
    run_segmentation()

    elapsed = time.time() - total_start
    banner(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print("""
  Generated artefacts
  ───────────────────
  data/transactions.csv           → raw synthetic transactions
  data/features.csv               → engineered feature matrix
  models/best_model.pkl           → best trained model
  models/feature_importance.csv   → top features
  models/training_report.json     → metrics & comparison
  models/test_predictions.csv     → test-set predictions
  models/segmented_customers.csv  → customers with segments

  Next steps
  ──────────
  Run the API:        uvicorn api.main:app --reload --port 8000
  Run the Streamlit:  streamlit run app/streamlit_app.py
""")


if __name__ == "__main__":
    main()
