from backend.optimizer.fleet_router import optimize


def test_optimize_assigns_heaviest_first_and_respects_capacity():
    """Heavier delivery gets the higher-perf truck; lighter one uses the next best."""
    deliveries = [
        {"delivery_id": 1, "weight_kg": 100.0},
        {"delivery_id": 2, "weight_kg": 500.0},
    ]
    trucks = [
        {"truck_id": 10, "capacity_kg": 200.0, "driver_perf_score": 0.5},
        {"truck_id": 20, "capacity_kg": 600.0, "driver_perf_score": 0.9},
        {"truck_id": 30, "capacity_kg": 600.0, "driver_perf_score": 0.8},
    ]
    out = optimize(deliveries, trucks)
    routes = {r["delivery_id"]: r["truck_id"] for r in out["routes"]}
    assert routes[2] == 20
    assert routes[1] == 30


def test_optimize_skips_when_no_truck_can_carry():
    """No route is emitted when every truck is under capacity for the load."""
    deliveries = [{"delivery_id": 1, "weight_kg": 9999.0}]
    trucks = [{"truck_id": 1, "capacity_kg": 100.0, "driver_perf_score": 0.99}]
    assert optimize(deliveries, trucks) == {"routes": []}
