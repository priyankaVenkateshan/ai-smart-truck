from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app


@pytest.fixture
def client():
    """FastAPI app wrapped for synchronous tests."""
    return TestClient(app)


def test_post_assign_returns_expected_json(client, monkeypatch):
    """POST /assign returns predictions and explanation when dependencies are mocked."""
    delivery = {
        "delivery_id": 1,
        "distance_km": 100.0,
        "hour_of_day": 12,
        "weight_kg": 500.0,
        "status": "pending",
    }
    trucks = [
        {
            "truck_id": 7,
            "capacity_kg": 1000.0,
            "driver_perf_score": 0.91,
            "fuel_eff_kmpl": 4.5,
            "driver_id": 3,
            "available": True,
        }
    ]

    monkeypatch.setattr(
        "backend.api.main.queries.get_delivery_by_id", lambda did: delivery if did == 1 else None
    )
    monkeypatch.setattr("backend.api.main.queries.get_available_trucks", lambda: trucks)
    monkeypatch.setattr(
        "backend.api.main.optimize",
        lambda d, t: {"routes": [{"delivery_id": 1, "truck_id": 7}]},
    )
    monkeypatch.setattr(
        "backend.api.main.queries.get_truck_with_driver",
        lambda tid: {"truck_id": tid, "driver_name": "Priya Driver", "driver_perf_score": 0.91, "fuel_eff_kmpl": 4.5},
    )

    class _P:
        def predict(self, features):
            return {
                "predicted_delay_min": 4.0,
                "predicted_fuel_l": 55.0,
                "predicted_eta_min": 180.0,
            }

    monkeypatch.setattr("backend.api.main.get_predictor", lambda: _P())
    monkeypatch.setattr("backend.api.main.explain", lambda a: "Manager-friendly summary.")
    monkeypatch.setattr("backend.api.main.queries.insert_assignment", lambda **k: 99)
    monkeypatch.setattr("backend.api.main.queries.assign_truck_to_delivery", lambda *a: None)

    r = client.post("/assign", json={"delivery_id": 1})
    assert r.status_code == 200
    data = r.json()
    assert data["truck_id"] == 7
    assert data["predicted_delay_min"] == pytest.approx(4.0)
    assert data["predicted_fuel_l"] == pytest.approx(55.0)
    assert data["predicted_eta_min"] == pytest.approx(180.0)
    assert data["explanation"] == "Manager-friendly summary."
    assert data["assignment_id"] == 99


def test_post_complete_returns_payload_and_schedules_retrain(client, monkeypatch):
    """POST /complete returns the mocked row and queues the retrain hook."""
    spy = MagicMock()
    monkeypatch.setattr("backend.api.main.retrain.check_and_maybe_retrain", spy)

    def fake_complete(did, eta, fuel, delay):
        return {
            "delivery_id": did,
            "actual_eta_min": eta,
            "actual_fuel_l": fuel,
            "actual_delay_min": delay,
            "on_time": True,
            "status": "completed",
        }

    monkeypatch.setattr("backend.api.main.queries.complete_delivery_transaction", fake_complete)

    r = client.post(
        "/deliveries/3/complete",
        json={"actual_eta_min": 120.0, "actual_fuel_l": 40.0, "actual_delay_min": 5.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["delivery_id"] == 3
    assert body["on_time"] is True
    spy.assert_called_once()
