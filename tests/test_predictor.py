from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.training.load_predictor import LoadPredictor


def test_predictor_handles_night_hour_and_zero_weight(monkeypatch, tmp_path):
    """Edge inputs should flow through without raising (models mocked)."""
    seq = iter(
        [
            MagicMock(predict=lambda X: np.array([3.5], dtype=np.float64)),
            MagicMock(predict=lambda X: np.array([8.0], dtype=np.float64)),
            MagicMock(predict=lambda X: np.array([120.0], dtype=np.float64)),
        ]
    )

    def fake_load(path):
        return next(seq)

    monkeypatch.setattr("backend.training.load_predictor.joblib.load", fake_load)

    p = LoadPredictor(models_dir=tmp_path)
    out = p.predict(
        {
            "distance_km": 50.0,
            "hour_of_day": 0.0,
            "weight_kg": 0.0,
            "driver_perf_score": 0.88,
            "fuel_eff_kmpl": 4.2,
        }
    )
    assert out["predicted_delay_min"] == pytest.approx(3.5)
    assert out["predicted_fuel_l"] == pytest.approx(8.0)
    assert out["predicted_eta_min"] == pytest.approx(120.0)


def test_predictor_chains_delay_into_eta_feature(monkeypatch, tmp_path):
    """ETA model receives the delay model output as its third input feature."""
    def make_delay_model():
        m = MagicMock()
        m.predict = lambda X: np.array([15.0], dtype=np.float64)
        return m

    fuel = MagicMock(predict=lambda X: np.array([22.0], dtype=np.float64))

    def make_eta_model():
        m = MagicMock()

        def _pred(X):
            assert X.shape == (1, 3)
            assert float(X[0, 2]) == 15.0
            return np.array([200.0], dtype=np.float64)

        m.predict = _pred
        return m

    seq = iter([make_delay_model(), fuel, make_eta_model()])

    monkeypatch.setattr("backend.training.load_predictor.joblib.load", lambda p: next(seq))

    p = LoadPredictor(models_dir=tmp_path)
    p.predict(
        {
            "distance_km": 80.0,
            "hour_of_day": 23.0,
            "weight_kg": 1.0,
            "driver_perf_score": 0.7,
            "fuel_eff_kmpl": 5.0,
        }
    )
