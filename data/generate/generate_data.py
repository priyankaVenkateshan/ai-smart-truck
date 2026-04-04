from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from faker import Faker
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)

fake = Faker()


@dataclass(frozen=True)
class Bounds:
    """Geographic bounding box for sampling lat/lng."""

    lat_min: float
    lat_max: float
    lng_min: float
    lng_max: float


INDIA_BOUNDS = Bounds(
    lat_min=8.0,
    lat_max=28.0,
    lng_min=68.0,
    lng_max=88.0,
)


def _rand_lat_lng(bounds: Bounds) -> tuple[float, float]:
    """Sample a random point inside `bounds`."""
    return (
        random.uniform(bounds.lat_min, bounds.lat_max),
        random.uniform(bounds.lng_min, bounds.lng_max),
    )


def _haversine_km(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    """Great-circle distance in km (synthetic stand-in for road distance)."""
    from math import asin, cos, radians, sin, sqrt

    r = 6371.0
    dlat = radians(b_lat - a_lat)
    dlng = radians(b_lng - a_lng)
    lat1 = radians(a_lat)
    lat2 = radians(b_lat)
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
    return 2 * r * asin(sqrt(h))


def generate_drivers(n: int) -> pd.DataFrame:
    """Build `n` synthetic driver rows."""
    rows = []
    for _ in range(n):
        rows.append(
            {
                "driver_name": fake.name(),
                "driver_perf_score": round(random.uniform(0.5, 0.95), 3),
            }
        )
    return pd.DataFrame(rows)


def generate_trucks(n: int, driver_ids: list[int]) -> pd.DataFrame:
    """Build `n` synthetic truck rows referencing `driver_ids`."""
    rows = []
    for _ in range(n):
        lat, lng = _rand_lat_lng(INDIA_BOUNDS)
        rows.append(
            {
                "driver_id": random.choice(driver_ids),
                "capacity_kg": round(random.uniform(800.0, 8000.0), 1),
                "fuel_eff_kmpl": round(random.uniform(2.5, 6.5), 2),
                "location_lat": round(lat, 6),
                "location_lng": round(lng, 6),
                "available": True,
            }
        )
    return pd.DataFrame(rows)


def generate_deliveries(n: int) -> pd.DataFrame:
    """Build `n` synthetic delivery rows."""
    rows = []
    for _ in range(n):
        o_lat, o_lng = _rand_lat_lng(INDIA_BOUNDS)
        d_lat, d_lng = _rand_lat_lng(INDIA_BOUNDS)
        distance_km = _haversine_km(o_lat, o_lng, d_lat, d_lng)
        weight_kg = max(1.0, random.gauss(mu=1200.0, sigma=700.0))

        rows.append(
            {
                "origin_lat": round(o_lat, 6),
                "origin_lng": round(o_lng, 6),
                "dest_lat": round(d_lat, 6),
                "dest_lng": round(d_lng, 6),
                "distance_km": round(max(distance_km, 1.0), 3),
                "weight_kg": round(min(weight_kg, 12000.0), 1),
                "hour_of_day": random.randint(0, 23),
                "status": "pending",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Truncate core tables and reload 50 trucks + 500 deliveries."""
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set. Create a .env file (see .env.example).")

    engine = create_engine(db_url, future=True)

    # Create fresh drivers/trucks/deliveries for repeatable demos.
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE assignments RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE deliveries RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE trucks RESTART IDENTITY CASCADE"))
        conn.execute(text("TRUNCATE TABLE drivers RESTART IDENTITY CASCADE"))

    drivers = generate_drivers(50)
    with engine.begin() as conn:
        drivers.to_sql("drivers", conn, if_exists="append", index=False)
        driver_ids = [row[0] for row in conn.execute(text("SELECT driver_id FROM drivers")).all()]

        trucks = generate_trucks(50, driver_ids)
        trucks.to_sql("trucks", conn, if_exists="append", index=False)

        deliveries = generate_deliveries(500)
        deliveries.to_sql("deliveries", conn, if_exists="append", index=False)

        trucks_count = conn.execute(text("SELECT COUNT(*) FROM trucks")).scalar_one()
        deliveries_count = conn.execute(text("SELECT COUNT(*) FROM deliveries")).scalar_one()

    log.info("Loaded synthetic data. trucks=%s, deliveries=%s", trucks_count, deliveries_count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
