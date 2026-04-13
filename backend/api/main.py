from __future__ import annotations

import os
from math import asin, cos, radians, sin, sqrt

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

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
    delivery_id: int | None = Field(default=None, ge=1)
    weight_kg: float | None = Field(default=None, ge=0)
    origin_city: str | None = None
    destination_city: str | None = None
    hour_of_day: int | None = Field(default=None, ge=0, le=23)

    @model_validator(mode="after")
    def _validate_mode(self):
        if self.delivery_id is not None:
            return self
        missing = [
            k
            for k in ("weight_kg", "origin_city", "destination_city", "hour_of_day")
            if getattr(self, k) in (None, "")
        ]
        if missing:
            raise ValueError(f"Missing fields for shipment assignment: {', '.join(missing)}")
        return self


class AssignResponse(BaseModel):
    delivery_id: int
    truck_id: int
    driver_name: str | None = None
    predicted_delay_min: float
    predicted_fuel_l: float
    predicted_eta_min: float
    explanation: str
    assignment_id: int
    origin_lat: float | None = None
    origin_lng: float | None = None
    dest_lat: float | None = None
    dest_lng: float | None = None

class MatchRequest(BaseModel):
    weight_kg: float = Field(..., ge=0)
    origin_city: str
    destination_city: str
    hour_of_day: int = Field(..., ge=0, le=23)


class MatchTruck(BaseModel):
    truck_id: int
    driver_name: str | None = None
    driver_perf_score: float
    capacity_kg: float
    fuel_eff_kmpl: float
    predicted_delay_min: float
    predicted_fuel_l: float
    predicted_eta_min: float


class MatchResponse(BaseModel):
    delivery_id: int
    origin_lat: float
    origin_lng: float
    dest_lat: float
    dest_lng: float
    distance_km: float
    recommendations: list[MatchTruck]


class AssignSelectBody(BaseModel):
    delivery_id: int = Field(..., ge=1)
    truck_id: int = Field(..., ge=1)


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

@app.get("/stats")
def stats() -> dict:
    """Return top-bar dashboard stats."""
    fuel_price = float(os.getenv("FUEL_PRICE_PER_L", "105"))
    return queries.get_stats(fuel_price_per_l=fuel_price)


@app.post("/match", response_model=MatchResponse)
def match(body: MatchRequest) -> MatchResponse:
    """Create a delivery draft and return top-3 candidate trucks with predictions."""
    o_lat, o_lng = _geocode_city(body.origin_city)
    d_lat, d_lng = _geocode_city(body.destination_city)
    distance_km = _haversine_km(o_lat, o_lng, d_lat, d_lng)
    delivery = queries.create_delivery(
        origin_lat=o_lat,
        origin_lng=o_lng,
        dest_lat=d_lat,
        dest_lng=d_lng,
        distance_km=distance_km,
        weight_kg=float(body.weight_kg),
        hour_of_day=int(body.hour_of_day),
    )

    trucks = queries.get_available_trucks()
    candidates = [
        t for t in trucks if float(t["capacity_kg"]) >= float(body.weight_kg)
    ]
    if not candidates:
        raise HTTPException(status_code=400, detail="no available truck can carry this weight")

    candidates = sorted(candidates, key=lambda t: float(t["driver_perf_score"]), reverse=True)[:3]
    recs: list[MatchTruck] = []
    predictor = get_predictor()
    for t in candidates:
        full = queries.get_truck_with_driver(int(t["truck_id"])) or t
        features = {
            "distance_km": float(delivery["distance_km"]),
            "hour_of_day": float(delivery["hour_of_day"]),
            "weight_kg": float(delivery["weight_kg"]),
            "driver_perf_score": float(full["driver_perf_score"]),
            "fuel_eff_kmpl": float(full["fuel_eff_kmpl"]),
        }
        preds = predictor.predict(features)
        recs.append(
            MatchTruck(
                truck_id=int(full["truck_id"]),
                driver_name=full.get("driver_name"),
                driver_perf_score=float(full["driver_perf_score"]),
                capacity_kg=float(full["capacity_kg"]),
                fuel_eff_kmpl=float(full["fuel_eff_kmpl"]),
                predicted_delay_min=float(preds["predicted_delay_min"]),
                predicted_fuel_l=float(preds["predicted_fuel_l"]),
                predicted_eta_min=float(preds["predicted_eta_min"]),
            )
        )

    return MatchResponse(
        delivery_id=int(delivery["delivery_id"]),
        origin_lat=float(delivery["origin_lat"]),
        origin_lng=float(delivery["origin_lng"]),
        dest_lat=float(delivery["dest_lat"]),
        dest_lng=float(delivery["dest_lng"]),
        distance_km=float(delivery["distance_km"]),
        recommendations=recs,
    )


@app.post("/assign/select", response_model=AssignResponse)
def assign_select(body: AssignSelectBody) -> AssignResponse:
    """Finalize an assignment for a given delivery_id and chosen truck_id."""
    delivery = queries.get_delivery_by_id(body.delivery_id)
    if delivery is None:
        raise HTTPException(status_code=404, detail="delivery not found")

    truck_full = queries.get_truck_with_driver(int(body.truck_id))
    if not truck_full:
        raise HTTPException(status_code=404, detail="truck not found")
    if float(truck_full["capacity_kg"]) < float(delivery["weight_kg"]):
        raise HTTPException(status_code=400, detail="truck capacity is below shipment weight")

    features = {
        "distance_km": float(delivery["distance_km"]),
        "hour_of_day": float(delivery["hour_of_day"]),
        "weight_kg": float(delivery["weight_kg"]),
        "driver_perf_score": float(truck_full["driver_perf_score"]),
        "fuel_eff_kmpl": float(truck_full["fuel_eff_kmpl"]),
    }
    preds = get_predictor().predict(features)
    explanation = explain(
        {
            "truck_id": int(truck_full["truck_id"]),
            "delivery_id": int(delivery["delivery_id"]),
            "predicted_delay_min": float(preds["predicted_delay_min"]),
            "predicted_fuel_l": float(preds["predicted_fuel_l"]),
            "predicted_eta_min": float(preds["predicted_eta_min"]),
        }
    )

    assignment_id = queries.insert_assignment(
        delivery_id=int(delivery["delivery_id"]),
        truck_id=int(truck_full["truck_id"]),
        predicted_eta_min=float(preds["predicted_eta_min"]),
        predicted_delay_min=float(preds["predicted_delay_min"]),
        predicted_fuel_l=float(preds["predicted_fuel_l"]),
        explanation=explanation,
    )
    queries.assign_truck_to_delivery(int(delivery["delivery_id"]), int(truck_full["truck_id"]))

    return AssignResponse(
        delivery_id=int(delivery["delivery_id"]),
        truck_id=int(truck_full["truck_id"]),
        driver_name=truck_full.get("driver_name"),
        predicted_delay_min=float(preds["predicted_delay_min"]),
        predicted_fuel_l=float(preds["predicted_fuel_l"]),
        predicted_eta_min=float(preds["predicted_eta_min"]),
        explanation=explanation,
        assignment_id=int(assignment_id),
        origin_lat=float(delivery.get("origin_lat")) if delivery.get("origin_lat") is not None else None,
        origin_lng=float(delivery.get("origin_lng")) if delivery.get("origin_lng") is not None else None,
        dest_lat=float(delivery.get("dest_lat")) if delivery.get("dest_lat") is not None else None,
        dest_lng=float(delivery.get("dest_lng")) if delivery.get("dest_lng") is not None else None,
    )


@app.post("/assign", response_model=AssignResponse)
def assign(body: AssignRequest) -> AssignResponse:
    """Assign a truck, predict metrics, explain, persist assignment + delivery link."""
    if body.delivery_id is not None:
        delivery = queries.get_delivery_by_id(body.delivery_id)
        if delivery is None:
            raise HTTPException(status_code=404, detail="delivery not found")
    else:
        o_lat, o_lng = _geocode_city(str(body.origin_city))
        d_lat, d_lng = _geocode_city(str(body.destination_city))
        distance_km = _haversine_km(o_lat, o_lng, d_lat, d_lng)
        delivery = queries.create_delivery(
            origin_lat=o_lat,
            origin_lng=o_lng,
            dest_lat=d_lat,
            dest_lng=d_lng,
            distance_km=distance_km,
            weight_kg=float(body.weight_kg),
            hour_of_day=int(body.hour_of_day),
        )

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
    truck = next((t for t in trucks if int(t["truck_id"]) == truck_id), None) or {}
    truck_full = queries.get_truck_with_driver(truck_id)
    if not truck_full:
        raise HTTPException(status_code=500, detail="assigned truck not found")

    features = {
        "distance_km": float(delivery["distance_km"]),
        "hour_of_day": float(delivery["hour_of_day"]),
        "weight_kg": float(delivery["weight_kg"]),
        "driver_perf_score": float(truck_full["driver_perf_score"]),
        "fuel_eff_kmpl": float(truck_full["fuel_eff_kmpl"]),
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
        driver_name=truck_full.get("driver_name"),
        predicted_delay_min=preds["predicted_delay_min"],
        predicted_fuel_l=preds["predicted_fuel_l"],
        predicted_eta_min=preds["predicted_eta_min"],
        explanation=explanation,
        assignment_id=assignment_id,
        origin_lat=float(delivery.get("origin_lat")) if delivery.get("origin_lat") is not None else None,
        origin_lng=float(delivery.get("origin_lng")) if delivery.get("origin_lng") is not None else None,
        dest_lat=float(delivery.get("dest_lat")) if delivery.get("dest_lat") is not None else None,
        dest_lng=float(delivery.get("dest_lng")) if delivery.get("dest_lng") is not None else None,
    )


def _haversine_km(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    """Compute great-circle distance in kilometers."""
    r = 6371.0
    dlat = radians(b_lat - a_lat)
    dlng = radians(b_lng - a_lng)
    lat1 = radians(a_lat)
    lat2 = radians(b_lat)
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    return 2 * r * asin(sqrt(h))


_GEO_CACHE: dict[str, tuple[float, float]] = {}


def _geocode_city(city: str) -> tuple[float, float]:
    """Geocode a city name using Nominatim (OpenStreetMap)."""
    key = city.strip().lower()
    if not key:
        raise HTTPException(status_code=400, detail="origin_city/destination_city cannot be empty")
    if key in _GEO_CACHE:
        return _GEO_CACHE[key]

    import httpx

    try:
        with httpx.Client(timeout=15.0, headers={"User-Agent": "ai-smart-truck/1.0"}) as client:
            r = client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": city, "format": "json", "limit": 1},
            )
            r.raise_for_status()
            data = r.json()
        if not data:
            raise HTTPException(status_code=400, detail=f"Could not geocode city: {city}")
        lat = float(data[0]["lat"])
        lng = float(data[0]["lon"])
        _GEO_CACHE[key] = (lat, lng)
        return lat, lng
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail=f"Could not geocode city: {city}")


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
