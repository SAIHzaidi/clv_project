"""
predictor_local.py
------------------
Inference wrapper compatible with the pure-numpy models from train_local.py.
Drop-in replacement for predictor.py when sklearn/lgbm are unavailable.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.segmentation import SEGMENT_RECOMMENDATIONS

MODELS_DIR = Path("models")


class CLVPredictor:
    def __init__(self, model_dir: Union[str, Path] = MODELS_DIR):
        model_dir = Path(model_dir)
        with open(model_dir / "best_model.pkl",    "rb") as f: self.model = pickle.load(f)
        with open(model_dir / "feature_names.pkl", "rb") as f: self.feature_names = pickle.load(f)
        fi_path = model_dir / "feature_importance.csv"
        self._fi = pd.read_csv(fi_path) if fi_path.exists() else None

    def predict(self, features: Dict) -> Dict:
        X   = self._build_input(features)
        clv = float(np.expm1(self.model.predict(X)[0]))
        clv = max(0.0, round(clv, 2))

        if clv >= 300:   segment = "HIGH"
        elif clv >= 80:  segment = "MEDIUM"
        else:            segment = "LOW"

        rec = SEGMENT_RECOMMENDATIONS[segment]
        return {
            "predicted_clv"  : clv,
            "segment"        : segment,
            "segment_label"  : rec["label"],
            "segment_emoji"  : rec["emoji"],
            "recommendations": rec["recommended_actions"],
            "confidence_note": (
                "Prediction based on historical transaction patterns. "
                "Accuracy is highest for customers with ≥3 months of history."
            ),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        X   = df.reindex(columns=self.feature_names, fill_value=0).values.astype(float)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            col_means = np.nanmean(X, axis=0)
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        clv = np.expm1(self.model.predict(X)).clip(0).round(2)
        out = df.copy()
        out["predicted_clv"] = clv
        out["segment"]       = pd.cut(clv, bins=[-np.inf, 80, 300, np.inf],
                                      labels=["LOW","MEDIUM","HIGH"]).astype(str)
        out["segment_label"] = out["segment"].map(
            lambda s: SEGMENT_RECOMMENDATIONS.get(s, {}).get("label", s))
        return out

    def feature_importance(self, top_n: int = 15) -> List[Dict]:
        if self._fi is None:
            return []
        return self._fi.head(top_n).to_dict(orient="records")

    def _build_input(self, features: Dict) -> np.ndarray:
        row = np.array([features.get(f, 0) for f in self.feature_names], dtype=float)
        return row.reshape(1, -1)
