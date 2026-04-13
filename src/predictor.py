"""
predictor.py
------------
Lightweight inference wrapper consumed by both the FastAPI endpoint
and the Streamlit application.  Loads persisted artefacts once and
exposes a clean predict() interface.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.segmentation import assign_segments, SEGMENT_RECOMMENDATIONS

MODELS_DIR = Path("models")


class CLVPredictor:
    """
    Thread-safe singleton-style predictor.
    Load once at app startup; call predict() for each request.
    """

    def __init__(self, model_dir: Union[str, Path] = MODELS_DIR):
        model_dir = Path(model_dir)
        self.model         = joblib.load(model_dir / "best_model.pkl")
        self.feature_names = joblib.load(model_dir / "feature_names.pkl")
        self._fi           = self._load_fi(model_dir)

    @staticmethod
    def _load_fi(model_dir: Path) -> Optional[pd.DataFrame]:
        fi_path = model_dir / "feature_importance.csv"
        return pd.read_csv(fi_path) if fi_path.exists() else None

    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, features: Dict) -> Dict:
        """
        Predict CLV and segment for a single customer.

        Parameters
        ----------
        features : dict mapping feature names to scalar values

        Returns
        -------
        dict with keys: predicted_clv, segment, segment_label,
                        recommendations, confidence_note
        """
        X = self._build_input(features)
        clv_log = self.model.predict(X)[0]
        clv     = float(np.expm1(clv_log))
        clv     = max(0.0, round(clv, 2))

        # Segment from a single-value series
        clv_series = pd.Series([clv])
        # Use fixed thresholds derived from training distribution
        if clv >= 300:
            segment = "HIGH"
        elif clv >= 80:
            segment = "MEDIUM"
        else:
            segment = "LOW"

        rec = SEGMENT_RECOMMENDATIONS[segment]

        return {
            "predicted_clv"  : clv,
            "segment"        : segment,
            "segment_label"  : rec["label"],
            "segment_emoji"  : rec["emoji"],
            "recommendations": rec["recommended_actions"],
            "confidence_note": (
                "Prediction is based on historical transaction patterns. "
                "Accuracy is highest for customers with ≥3 months of history."
            ),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict CLV for a DataFrame of customers.
        Missing features are imputed with 0 (conservative default).
        """
        X = df.reindex(columns=self.feature_names, fill_value=0)
        clv_log         = self.model.predict(X)
        df = df.copy()
        df["predicted_clv"] = np.expm1(clv_log).clip(0).round(2)

        # Vectorised segmentation using training-calibrated thresholds
        df["segment"] = pd.cut(
            df["predicted_clv"],
            bins=[-np.inf, 80, 300, np.inf],
            labels=["LOW", "MEDIUM", "HIGH"],
        ).astype(str)

        df["segment_label"] = df["segment"].map(
            lambda s: SEGMENT_RECOMMENDATIONS.get(s, {}).get("label", s)
        )
        return df

    def feature_importance(self, top_n: int = 15) -> List[Dict]:
        if self._fi is None:
            return []
        return self._fi.head(top_n).to_dict(orient="records")

    # ── private ───────────────────────────────────────────────────────────────

    def _build_input(self, features: Dict) -> pd.DataFrame:
        """
        Convert a raw feature dict into a properly ordered DataFrame.
        Missing features default to 0 (conservative / neutral).
        """
        row = {f: features.get(f, 0) for f in self.feature_names}
        return pd.DataFrame([row])
