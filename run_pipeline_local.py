"""
run_pipeline_local.py
---------------------
Pipeline runner that works without sklearn/lightgbm.
Uses the pure numpy/scipy implementations in train_local.py.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def banner(title):
    print("\n" + "═"*60)
    print(f"  {title}")
    print("═"*60)

def main():
    t0 = time.time()

    banner("STEP 1 / 4 — DATA GENERATION")
    from src.data_generator import generate_dataset
    generate_dataset(save_path="data/transactions.csv")

    banner("STEP 2 / 4 — FEATURE ENGINEERING")
    from src.feature_engineering import build_feature_matrix
    features, _ = build_feature_matrix(path="data/transactions.csv")
    features.to_csv("data/features.csv", index=False)

    banner("STEP 3 / 4 — MODEL TRAINING")
    from src.train_local import train
    train()

    banner("STEP 4 / 4 — SEGMENTATION")
    from src.segmentation import run_segmentation
    run_segmentation()

    print(f"\n✓ Pipeline complete ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
