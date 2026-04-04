"""PostgreSQL helpers using psycopg2 (Wednesday plan)."""

from __future__ import annotations

import os

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor


def dsn_for_psycopg2() -> str:
    """Return a DSN string psycopg2 accepts (normalizes sqlalchemy-style +psycopg URLs)."""
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env.")
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url.removeprefix("postgresql+psycopg://")
    return url


def connect():
    """Open a new psycopg2 connection."""
    return psycopg2.connect(dsn_for_psycopg2())


def get_available_trucks() -> list[dict]:
    """Return available trucks with driver performance score (one row per truck)."""
    sql = """
        SELECT trucks.*, drivers.driver_perf_score
        FROM trucks
        JOIN drivers USING (driver_id)
        WHERE trucks.available = TRUE
    """
    conn = connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_delivery_by_id(delivery_id: int) -> dict | None:
    """Return one delivery row or None."""
    conn = connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM deliveries WHERE delivery_id = %s", (delivery_id,))
            row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_trucks_live() -> list[dict]:
    """All trucks with driver info, location, and availability."""
    sql = """
        SELECT trucks.*, drivers.driver_name, drivers.driver_perf_score
        FROM trucks
        JOIN drivers USING (driver_id)
        ORDER BY trucks.truck_id
    """
    conn = connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def insert_assignment(
    delivery_id: int,
    truck_id: int,
    predicted_eta_min: float,
    predicted_delay_min: float,
    predicted_fuel_l: float,
    explanation: str,
) -> int:
    """Insert a row into assignments; returns assignment_id."""
    sql = """
        INSERT INTO assignments (
            delivery_id, truck_id,
            predicted_eta_min, predicted_delay_min, predicted_fuel_l,
            explanation
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING assignment_id
    """
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    delivery_id,
                    truck_id,
                    predicted_eta_min,
                    predicted_delay_min,
                    predicted_fuel_l,
                    explanation,
                ),
            )
            assignment_id = cur.fetchone()[0]
        conn.commit()
        return int(assignment_id)
    finally:
        conn.close()


def assign_truck_to_delivery(delivery_id: int, truck_id: int) -> None:
    """Persist chosen truck on the delivery row after /assign."""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE deliveries
                SET assigned_truck_id = %s, status = %s
                WHERE delivery_id = %s
                """,
                (truck_id, "assigned", delivery_id),
            )
        conn.commit()
    finally:
        conn.close()


def count_completions_since_retrain() -> int:
    """Deliveries with completed_at after last retrain (or all completed if never retrained)."""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT last_retrain_at FROM retrain_state WHERE id = 1")
            row = cur.fetchone()
            last = row[0] if row else None
            if last is None:
                cur.execute(
                    "SELECT COUNT(*) FROM deliveries WHERE completed_at IS NOT NULL"
                )
            else:
                cur.execute(
                    """
                    SELECT COUNT(*) FROM deliveries
                    WHERE completed_at IS NOT NULL AND completed_at > %s
                    """,
                    (last,),
                )
            return int(cur.fetchone()[0])
    finally:
        conn.close()


def mark_retrain_done_now() -> None:
    """Set `retrain_state.last_retrain_at` to the current time."""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO retrain_state (id, last_retrain_at)
                VALUES (1, NOW())
                ON CONFLICT (id) DO UPDATE SET last_retrain_at = EXCLUDED.last_retrain_at
                """
            )
        conn.commit()
    finally:
        conn.close()


def complete_delivery_transaction(
    delivery_id: int,
    actual_eta_min: float,
    actual_fuel_l: float,
    actual_delay_min: float,
) -> dict:
    """
    Set actuals, on_time, completed_at; update driver's perf score.
    Returns updated delivery as dict.
    Raises LookupError if delivery missing, RuntimeError if already completed / invalid.
    """
    conn = connect()
    try:
        conn.autocommit = False
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM deliveries WHERE delivery_id = %s FOR UPDATE",
                (delivery_id,),
            )
            drow = cur.fetchone()
            if not drow:
                raise LookupError("delivery not found")
            d = dict(drow)
            if d.get("completed_at") is not None:
                raise RuntimeError("already completed")

            truck_id = d.get("assigned_truck_id")
            if truck_id is None:
                cur.execute(
                    """
                    SELECT truck_id FROM assignments
                    WHERE delivery_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (delivery_id,),
                )
                r = cur.fetchone()
                truck_id = int(r["truck_id"]) if r else None

            if truck_id is None:
                raise RuntimeError("delivery has no assigned truck")

            cur.execute(
                """
                SELECT predicted_delay_min
                FROM assignments
                WHERE delivery_id = %s AND predicted_delay_min IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (delivery_id,),
            )
            prow = cur.fetchone()
            predicted_delay = (
                float(prow["predicted_delay_min"])
                if prow and prow["predicted_delay_min"] is not None
                else None
            )

            if predicted_delay is not None:
                on_time = float(actual_delay_min) <= predicted_delay
            else:
                on_time = float(actual_delay_min) <= 30.0

            cur.execute(
                """
                UPDATE deliveries
                SET actual_eta_min = %s,
                    actual_fuel_l = %s,
                    actual_delay_min = %s,
                    on_time = %s,
                    completed_at = NOW(),
                    status = %s
                WHERE delivery_id = %s
                RETURNING *
                """,
                (
                    actual_eta_min,
                    actual_fuel_l,
                    actual_delay_min,
                    on_time,
                    "completed",
                    delivery_id,
                ),
            )
            updated = dict(cur.fetchone())

            cur.execute(
                "SELECT driver_id FROM trucks WHERE truck_id = %s",
                (truck_id,),
            )
            trow = cur.fetchone()
            if not trow:
                raise RuntimeError("truck not found")
            driver_id = int(trow["driver_id"])

            cur.execute(
                "SELECT driver_perf_score FROM drivers WHERE driver_id = %s FOR UPDATE",
                (driver_id,),
            )
            dr = cur.fetchone()
            old_score = float(dr["driver_perf_score"])
            reward = 1.0 if on_time else 0.0
            new_score = 0.9 * old_score + 0.1 * reward
            cur.execute(
                "UPDATE drivers SET driver_perf_score = %s WHERE driver_id = %s",
                (new_score, driver_id),
            )

        conn.commit()
        return updated
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_recent_assignments(limit: int = 20) -> list[dict]:
    """Recent assignment rows, newest first (includes delivery on_time when known)."""
    conn = connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT a.*, d.on_time AS delivery_on_time
                FROM assignments a
                LEFT JOIN deliveries d ON d.delivery_id = a.delivery_id
                ORDER BY a.created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
