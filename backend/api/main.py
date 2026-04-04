from __future__ import annotations

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.db import queries
from backend.training import retrain
from backend.explainer.claude_explainer import explain
from backend.optimizer.fleet_router import optimize
from backend.training.load_predictor import LoadPredictor

app = FastAPI(title="AI Smart Truck API", version="0.2.0")

_predictor: LoadPredictor | None = None


def get_predictor() -> LoadPredictor:
    """Lazily construct and cache the shared `LoadPredictor` instance."""
    global _predictor
    if _predictor is None:
        _predictor = LoadPredictor()
    return _predictor


class AssignRequest(BaseModel):
    delivery_id: int = Field(..., ge=1)


class AssignResponse(BaseModel):
    delivery_id: int
    truck_id: int
    predicted_delay_min: float
    predicted_fuel_l: float
    predicted_eta_min: float
    explanation: str
    assignment_id: int


class DeliveryCompleteBody(BaseModel):
    actual_eta_min: float = Field(..., ge=0)
    actual_fuel_l: float = Field(..., ge=0)
    actual_delay_min: float = Field(..., ge=0)


@app.get("/health")
async def health() -> dict:
    """Return a simple OK payload for uptime checks."""
    return {"status": "ok"}


@app.get("/trucks/live")
def trucks_live() -> list[dict]:
    """All trucks with location and availability (plus driver fields)."""
    return queries.get_trucks_live()


@app.get("/api/assignments/recent")
def assignments_recent(limit: int = 20) -> list[dict]:
    """Return the newest assignment rows (optional `limit`, default 20)."""
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    return queries.get_recent_assignments(limit)


@app.post("/assign", response_model=AssignResponse)
def assign(body: AssignRequest) -> AssignResponse:
    """Assign a truck, predict metrics, explain, persist assignment + delivery link."""
    delivery = queries.get_delivery_by_id(body.delivery_id)
    if delivery is None:
        raise HTTPException(status_code=404, detail="delivery not found")

    trucks = queries.get_available_trucks()
    if not trucks:
        raise HTTPException(status_code=400, detail="no available trucks")

    delivery_for_opt = {
        "delivery_id": int(delivery["delivery_id"]),
        "weight_kg": float(delivery["weight_kg"]),
    }
    opt = optimize([delivery_for_opt], trucks)
    if not opt.get("routes"):
        raise HTTPException(
            status_code=400,
            detail="no truck with enough capacity for this delivery",
        )

    route = opt["routes"][0]
    truck_id = int(route["truck_id"])
    truck = next((t for t in trucks if int(t["truck_id"]) == truck_id), None)
    if truck is None:
        raise HTTPException(status_code=500, detail="assigned truck not found in pool")

    features = {
        "distance_km": float(delivery["distance_km"]),
        "hour_of_day": float(delivery["hour_of_day"]),
        "weight_kg": float(delivery["weight_kg"]),
        "driver_perf_score": float(truck["driver_perf_score"]),
        "fuel_eff_kmpl": float(truck["fuel_eff_kmpl"]),
    }
    preds = get_predictor().predict(features)

    assignment_ctx = {
        "truck_id": truck_id,
        "delivery_id": int(delivery["delivery_id"]),
        "predicted_delay_min": preds["predicted_delay_min"],
        "predicted_fuel_l": preds["predicted_fuel_l"],
        "predicted_eta_min": preds["predicted_eta_min"],
    }
    explanation = explain(assignment_ctx)

    assignment_id = queries.insert_assignment(
        delivery_id=int(delivery["delivery_id"]),
        truck_id=truck_id,
        predicted_eta_min=preds["predicted_eta_min"],
        predicted_delay_min=preds["predicted_delay_min"],
        predicted_fuel_l=preds["predicted_fuel_l"],
        explanation=explanation,
    )
    queries.assign_truck_to_delivery(int(delivery["delivery_id"]), truck_id)

    return AssignResponse(
        delivery_id=int(delivery["delivery_id"]),
        truck_id=truck_id,
        predicted_delay_min=preds["predicted_delay_min"],
        predicted_fuel_l=preds["predicted_fuel_l"],
        predicted_eta_min=preds["predicted_eta_min"],
        explanation=explanation,
        assignment_id=assignment_id,
    )


@app.post("/deliveries/{delivery_id}/complete")
def complete_delivery(
    delivery_id: int,
    body: DeliveryCompleteBody,
    background_tasks: BackgroundTasks,
) -> dict:
    """Record actuals, update driver score, enqueue optional auto-retrain."""
    try:
        updated = queries.complete_delivery_transaction(
            delivery_id,
            body.actual_eta_min,
            body.actual_fuel_l,
            body.actual_delay_min,
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="delivery not found")
    except RuntimeError as e:
        msg = str(e)
        if "already completed" in msg:
            raise HTTPException(status_code=400, detail=msg)
        if "no assigned truck" in msg:
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=400, detail=msg)

    background_tasks.add_task(retrain.check_and_maybe_retrain)
    return updated
