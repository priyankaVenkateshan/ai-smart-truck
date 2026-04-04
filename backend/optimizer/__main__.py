"""Smoke test: 2 deliveries + 5 trucks from DB → greedy routes."""

from __future__ import annotations

import logging

from psycopg2.extras import RealDictCursor

from backend.db.queries import connect
from backend.optimizer.fleet_router import optimize

log = logging.getLogger(__name__)


def main() -> None:
    """Load sample rows from Postgres and print the optimizer result."""
    conn = connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT delivery_id, weight_kg
                FROM deliveries
                ORDER BY delivery_id
                LIMIT 2
                """
            )
            deliveries = [dict(r) for r in cur.fetchall()]

            cur.execute(
                """
                SELECT trucks.*, drivers.driver_perf_score
                FROM trucks
                JOIN drivers USING (driver_id)
                ORDER BY truck_id
                LIMIT 5
                """
            )
            trucks = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    result = optimize(deliveries, trucks)
    log.info("optimizer result: %s", result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
