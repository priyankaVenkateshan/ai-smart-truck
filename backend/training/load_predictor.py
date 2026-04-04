"""Load trained XGBoost models once and produce delay, fuel, and ETA predictions."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

# Keep in sync with `train_models.py` column order for model inputs.
_DELAY_ORDER = ("distance_km", "hour_of_day", "weight_kg", "driver_perf_score")
_FUEL_ORDER = ("distance_km", "weight_kg", "fuel_eff_kmpl")


class LoadPredictor:
    """Loads `delay_model`, `fuel_model`, and `eta_model` from `models/`."""

    def __init__(self, models_dir: Path | None = None) -> None:
        """Load the three `.joblib` models from `models_dir` (default: repo `models/`)."""
        root = models_dir or Path(__file__).resolve().parent.parent.parent / "models"
        self._delay = joblib.load(root / "delay_model.joblib")
        self._fuel = joblib.load(root / "fuel_model.joblib")
        self._eta = joblib.load(root / "eta_model.joblib")

    def predict(self, features: dict) -> dict[str, float]:
        """
        features: distance_km, hour_of_day, weight_kg, driver_perf_score, fuel_eff_kmpl
        ETA uses predicted delay in place of actual_delay_min (unknown at inference).
        """
        X_delay = np.array([[float(features[k]) for k in _DELAY_ORDER]], dtype=np.float64)
        delay = float(self._delay.predict(X_delay)[0])

        X_fuel = np.array([[float(features[k]) for k in _FUEL_ORDER]], dtype=np.float64)
        fuel = float(self._fuel.predict(X_fuel)[0])

        X_eta = np.array(
            [[float(features["distance_km"]), float(features["driver_perf_score"]), delay]],
            dtype=np.float64,
        )
        eta = float(self._eta.predict(X_eta)[0])

        return {
            "predicted_delay_min": delay,
            "predicted_fuel_l": fuel,
            "predicted_eta_min": eta,
        }


__all__ = ["LoadPredictor"]
