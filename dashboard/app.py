"""
Friday dashboard: live trucks + recent assignments (FastAPI backend).
"""

from __future__ import annotations

import os
import time
from datetime import timedelta
from typing import Any

import folium
import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
DELAY_MAE_DEFAULT = os.getenv("DELAY_MODEL_MAE", "4.406")
ON_TIME_FALLBACK = os.getenv("DASHBOARD_ON_TIME_PCT")


def _fetch_json(path: str) -> Any:
    """GET a JSON value from the API under `API_BASE`."""
    url = f"{API_BASE}{path}"
    with httpx.Client(timeout=15.0) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.json()


def _on_time_pct(assignments: list[dict]) -> float | None:
    """Compute on-time percentage from joined `delivery_on_time` or env fallback."""
    rated = [a for a in assignments if a.get("delivery_on_time") is not None]
    if rated:
        ok = sum(1 for a in rated if a["delivery_on_time"] is True)
        return 100.0 * ok / len(rated)
    if ON_TIME_FALLBACK:
        try:
            return float(ON_TIME_FALLBACK)
        except ValueError:
            return None
    return None


def _avg_fuel(assignments: list[dict]) -> float | None:
    """Mean predicted fuel (liters) over assignments with that field set."""
    fuels = [a["predicted_fuel_l"] for a in assignments if a.get("predicted_fuel_l") is not None]
    if not fuels:
        return None
    return float(sum(fuels) / len(fuels))


@st.fragment(run_every=timedelta(seconds=30))
def _dashboard_fragment() -> None:
    """Fetch API data and render map, metrics, and selectable assignment table."""
    st.session_state["last_refresh"] = time.time()

    try:
        trucks = _fetch_json("/trucks/live")
        assignments = _fetch_json("/api/assignments/recent?limit=50")
    except Exception as e:
        st.error(f"Could not reach API at {API_BASE}: {e}")
        st.caption("Start the API: `py -3.11 -m uvicorn backend.api.main:app --reload`")
        return

    if not isinstance(trucks, list):
        trucks = []
    if not isinstance(assignments, list):
        assignments = []

    col_map, col_metrics = st.columns([1.6, 1.0], gap="large")

    with col_map:
        st.subheader("Live trucks")
        if not trucks:
            st.info("No trucks returned from /trucks/live.")
        else:
            lats = [float(t["location_lat"]) for t in trucks]
            lngs = [float(t["location_lng"]) for t in trucks]
            center = [sum(lats) / len(lats), sum(lngs) / len(lngs)]
            zoom = 5 if max(lats) - min(lats) > 5 else 7

            m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")
            for t in trucks:
                avail = bool(t.get("available", True))
                color = "green" if avail else "red"
                name = t.get("driver_name") or "—"
                tid = t.get("truck_id", "?")
                folium.CircleMarker(
                    location=[float(t["location_lat"]), float(t["location_lng"])],
                    radius=8,
                    color=color,
                    weight=2,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.65,
                    popup=folium.Popup(f"Truck {tid}<br/>{name}", max_width=220),
                ).add_to(m)

            st_folium(m, width=None, height=420, key="truck_map", returned_objects=[])

    with col_metrics:
        st.subheader("Metrics")
        otp = _on_time_pct(assignments)
        st.metric(
            "On-time %",
            f"{otp:.1f}%" if otp is not None else "—",
            help="Share of recent assignments whose delivery has on_time set (completed). "
            "If none yet, set DASHBOARD_ON_TIME_PCT in .env for a demo value.",
        )
        af = _avg_fuel(assignments)
        st.metric(
            "Avg predicted fuel (L)",
            f"{af:.1f}" if af is not None else "—",
            help="Mean of predicted_fuel_l over recent assignments.",
        )
        try:
            mae_val = float(DELAY_MAE_DEFAULT)
            st.metric("Delay model MAE (min)", f"{mae_val:.3f}", help="From training; override with DELAY_MODEL_MAE in .env.")
        except ValueError:
            st.metric("Delay model MAE (min)", str(DELAY_MAE_DEFAULT))

    st.subheader("Recent assignments")
    if not assignments:
        st.caption("No assignments yet. POST /assign from the API docs to create one.")
        return

    df = pd.DataFrame(assignments)
    display_cols = [
        c
        for c in (
            "assignment_id",
            "delivery_id",
            "truck_id",
            "predicted_delay_min",
            "predicted_fuel_l",
            "predicted_eta_min",
            "delivery_on_time",
            "created_at",
        )
        if c in df.columns
    ]
    df_view = df[display_cols].copy()

    event = st.dataframe(
        df_view,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="assignments_table",
        hide_index=True,
    )

    rows: list[int] = []
    if isinstance(event, dict):
        rows = list(event.get("selection", {}).get("rows", []))
    elif event is not None:
        sel = getattr(event, "selection", None)
        if isinstance(sel, dict):
            rows = list(sel.get("rows", []))
        elif sel is not None:
            rows = list(getattr(sel, "rows", []) or [])

    if rows:
        row_idx = int(rows[0])
        if 0 <= row_idx < len(assignments):
            st.info(assignments[row_idx].get("explanation") or "(No explanation text.)")


def main() -> None:
    """Configure the Streamlit page and mount the auto-refreshing dashboard fragment."""
    st.set_page_config(page_title="AI Smart Load Matching", layout="wide")
    st.title("AI Smart Load Matching")
    st.caption(
        f"API: `{API_BASE}` · Auto-refresh every 30s via `@st.fragment(run_every=…)`; "
        "`st.session_state['last_refresh']` updates each tick."
    )

    _dashboard_fragment()


main()
