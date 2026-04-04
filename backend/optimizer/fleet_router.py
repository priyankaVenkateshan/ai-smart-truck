"""Greedy load-to-truck assignment (no external routing API)."""

from __future__ import annotations


def optimize(deliveries: list[dict], trucks: list[dict]) -> dict:
    """
    For each delivery, assign a truck with capacity_kg >= weight_kg and the highest
    driver_perf_score. Each truck is used at most once per call.
    """
    # Heavier deliveries first — simple greedy ordering
    pending = sorted(
        deliveries,
        key=lambda d: float(d["weight_kg"]),
        reverse=True,
    )
    pool = list(trucks)
    routes: list[dict] = []

    for d in pending:
        w = float(d["weight_kg"])
        did = int(d["delivery_id"])
        candidates = [t for t in pool if float(t["capacity_kg"]) >= w]
        if not candidates:
            continue
        best = max(candidates, key=lambda t: float(t["driver_perf_score"]))
        routes.append({"delivery_id": did, "truck_id": int(best["truck_id"])})
        pool.remove(best)

    return {"routes": routes}
