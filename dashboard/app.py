"""AI Smart Load Matching dashboard (professional business UI)."""

from __future__ import annotations

import html as html_lib
import os
import time
from datetime import timedelta
from statistics import mean
from typing import Any

import folium
import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
FUEL_PRICE_PER_L = float(os.getenv("FUEL_PRICE_PER_L", "105"))

GREEN = "#00C851"
RED = "#ff4444"


def _inject_dark_css() -> None:
    """Inject a dark, business-style theme via CSS."""
    st.markdown(
        f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
  .stApp {{
    background: radial-gradient(1200px 600px at 20% 0%, #182032 0%, #0B1020 55%, #070A12 100%);
    color: #E8EAF0;
  }}
  .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }}
  h1, h2, h3 {{ color: #F1F3F7; }}
  .muted {{ color: #A9B1C7; }}
  .card {{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px 14px;
  }}
  .pill {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 12px;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.06);
  }}
  .ok {{ color: {GREEN}; }}
  .bad {{ color: {RED}; }}
  .stMetric {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 10px;
  }}
  [data-testid="stForm"] {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 14px 14px 10px 14px;
  }}
  .stDataFrame {{
    background: rgba(255,255,255,0.02);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.06);
  }}

  .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
  .stTabs [data-baseweb="tab"] {{
    background: rgba(255,255,255,0.04);
    border-radius: 10px 10px 0 0;
    padding: 8px 20px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.1em;
  }}
  .stTabs [aria-selected="true"] {{
    background: rgba(255,255,255,0.09);
    border-top: 2px solid #00C851;
  }}

  .sec-h {{
    font-family: 'Space Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-size: 13px;
    color: #97A3BE;
    border-left: 3px solid #00C851;
    padding-left: 10px;
    margin: 0.5rem 0 0.75rem 0;
  }}

  /* Command-center metrics */
  .fleet-metrics-shell {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 4px;
  }}
  .cc-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
  }}
  .cc-hero {{
    background: #0d1117;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 16px 16px 14px 16px;
    position: relative;
    overflow: hidden;
  }}
  .cc-hero.available {{
    border-top: 2px solid {GREEN};
  }}
  .cc-hero.fleet {{
    border-top: 2px solid rgba(255,255,255,0.22);
  }}
  .cc-label {{
    font-family: "Space Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    letter-spacing: 0.14em;
    font-size: 12px;
    color: #C8D1E6;
  }}
  .cc-number {{
    font-family: "Space Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-weight: 700;
    font-size: 52px;
    line-height: 1.05;
    color: #F6F7FB;
    margin-top: 6px;
  }}
  .cc-sub {{
    font-family: "Space Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    color: #97A3BE;
    font-size: 13px;
    margin-top: 2px;
  }}
  .cc-bar {{
    height: 8px;
    background: rgba(255,255,255,0.10);
    border-radius: 999px;
    margin-top: 10px;
    overflow: hidden;
  }}
  .cc-bar > div {{
    height: 100%;
    background: {GREEN};
    width: 50%;
  }}
  .cc-types {{
    margin-top: 12px;
    background: #0d1117;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 14px 14px 14px 14px;
  }}
  .cc-type-row {{
    display: grid;
    grid-template-columns: 120px 1fr 80px 80px;
    gap: 10px;
    align-items: center;
    padding: 10px 8px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.03);
    margin-top: 10px;
  }}
  .cc-pill {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: "Space Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    letter-spacing: 0.12em;
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.06);
  }}
  .cc-mini {{
    height: 6px;
    background: rgba(255,255,255,0.10);
    border-radius: 999px;
    overflow: hidden;
    margin-top: 6px;
  }}
  .cc-mini > div {{
    height: 100%;
    width: 50%;
  }}

  .type-pill-rec {{
    display: inline-block;
    font-family: "Space Mono", ui-monospace, monospace;
    font-size: 10px;
    letter-spacing: 0.08em;
    padding: 3px 8px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(255,255,255,0.08);
    color: #C8D1E6;
    margin-left: 8px;
    vertical-align: middle;
  }}

  div[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 12px;
    margin-bottom: 12px !important;
    border-color: rgba(255,255,255,0.12) !important;
    background: rgba(255,255,255,0.03);
    padding: 4px 8px 8px 8px;
  }}

  .fleet-legend {{
    display: flex;
    gap: 24px;
    align-items: center;
    margin-top: 10px;
    font-family: "Space Mono", ui-monospace, monospace;
    font-size: 12px;
    color: #A9B1C7;
  }}
  .fleet-legend span {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }}
  .dot-g {{ width: 10px; height: 10px; border-radius: 50%; background: {GREEN}; display: inline-block; }}
  .dot-r {{ width: 10px; height: 10px; border-radius: 50%; background: {RED}; display: inline-block; }}

  .ops-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 8px;
  }}
  .ops-table th, .ops-table td {{
    border: 1px solid rgba(255,255,255,0.08);
    padding: 8px 10px;
    text-align: left;
  }}
  .ops-table th {{
    font-family: "Space Mono", ui-monospace, monospace;
    letter-spacing: 0.06em;
    font-size: 11px;
    color: #97A3BE;
    text-transform: uppercase;
  }}
</style>
""",
        unsafe_allow_html=True,
    )


def _section_header(title: str) -> None:
    st.markdown(f'<div class="sec-h">{html_lib.escape(title)}</div>', unsafe_allow_html=True)


def _fetch_json(path: str) -> Any:
    """GET a JSON value from the API under `API_BASE`."""
    url = f"{API_BASE}{path}"
    with httpx.Client(timeout=15.0) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.json()


def _post_match(payload: dict) -> dict:
    """POST shipment details to /match and return top-3 recommendations."""
    url = f"{API_BASE}/match"
    with httpx.Client(timeout=30.0) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


def _post_assign_select(payload: dict) -> dict:
    """POST delivery_id + truck_id to finalize a chosen recommendation."""
    url = f"{API_BASE}/assign/select"
    with httpx.Client(timeout=30.0) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


def _fleet_map(trucks: list[dict]) -> folium.Map:
    """Create a map showing all trucks (India view)."""
    m = folium.Map(location=[22.5, 79.0], zoom_start=5, tiles="CartoDB dark_matter")
    for t in trucks:
        avail = bool(t.get("available", True))
        color = GREEN if avail else RED
        tid = t.get("truck_id", "?")
        name = t.get("driver_name") or "—"
        cap = t.get("capacity_kg")
        folium.CircleMarker(
            location=[float(t["location_lat"]), float(t["location_lng"])],
            radius=7,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(
                f"<b>Truck {tid}</b><br/>{name}<br/>Capacity: {cap} kg<br/>"
                f"Status: {'Available' if avail else 'Busy'}",
                max_width=260,
            ),
        ).add_to(m)
    return m


def _fleet_metrics_html(
    *,
    total: int,
    available: int,
    busy: int,
    pct: int,
    busy_pct: int,
    buckets: dict[str, dict[str, int]],
    order: list[str],
    type_colors: dict[str, str],
) -> str:
    """Single HTML string for fleet metrics + by-type rows (one markdown render)."""
    type_rows_parts: list[str] = []
    for typ in order:
        if typ not in buckets:
            continue
        tot = buckets[typ]["total"]
        av = buckets[typ]["available"]
        tpct = 0 if tot == 0 else int(round(100 * av / tot))
        accent = type_colors.get(typ, "#4FC3F7")
        typ_esc = html_lib.escape(typ.upper())
        type_rows_parts.append(
            f"""
<div class="cc-type-row" style="border-top: 2px solid {accent};">
  <div><span class="cc-pill" style="color:{accent}; border-color: rgba(255,255,255,0.16);">{typ_esc}</span></div>
  <div>
    <div class="cc-sub">AVAILABLE <span style="color:{GREEN}; font-weight:700;">{av}</span> / TOTAL {tot}</div>
    <div class="cc-mini"><div style="width:{tpct}%; background:{accent};"></div></div>
  </div>
  <div class="cc-sub" style="text-align:right;">AVAIL</div>
  <div class="cc-sub" style="text-align:right;">{tpct}%</div>
</div>"""
        )
    type_rows_html = "".join(type_rows_parts)

    return f"""
<div class="fleet-metrics-shell">
  <div class="cc-grid">
    <div class="cc-hero available">
      <div class="cc-label">AVAILABLE TRUCKS</div>
      <div class="cc-number">{available}</div>
      <div class="cc-sub">{pct}% of fleet available · {busy} busy</div>
      <div class="cc-bar"><div style="width:{pct}%;"></div></div>
    </div>
    <div class="cc-hero fleet">
      <div class="cc-label">TOTAL FLEET</div>
      <div class="cc-number">{total}</div>
      <div class="cc-sub">{busy} on active routes · {available} ready</div>
      <div class="cc-bar"><div style="width:{busy_pct}%; background: rgba(255,255,255,0.25);"></div></div>
    </div>
  </div>
  <div class="cc-types">
    <div class="cc-label">BY TRUCK TYPE</div>
    {type_rows_html}
  </div>
</div>
"""


def _selection_rows_from_dataframe_event(event: Any) -> list[int]:
    rows: list[int] = []
    if isinstance(event, dict):
        rows = list(event.get("selection", {}).get("rows", []))
    elif event is not None:
        sel = getattr(event, "selection", None)
        if isinstance(sel, dict):
            rows = list(sel.get("rows", []))
        elif sel is not None:
            rows = list(getattr(sel, "rows", []) or [])
    return rows


@st.fragment(run_every=timedelta(seconds=30))
def _dashboard_fragment() -> None:
    """Render the full dashboard (auto-refreshing data layer)."""
    st.session_state["last_refresh"] = time.time()

    try:
        trucks = _fetch_json("/trucks/live")
        assignments = _fetch_json("/api/assignments/recent?limit=50")
        past_assignments = _fetch_json("/api/assignments/recent?limit=200")
    except Exception as e:
        st.error(f"Could not reach API at {API_BASE}: {e}")
        st.caption("Start the API: `py -3.11 -m uvicorn backend.api.main:app --reload`")
        return

    if not isinstance(trucks, list):
        trucks = []
    if not isinstance(assignments, list):
        assignments = []
    if not isinstance(past_assignments, list):
        past_assignments = []

    truck_type_by_id: dict[int, str] = {}
    for t in trucks:
        tid = t.get("truck_id")
        if tid is not None:
            truck_type_by_id[int(tid)] = str(t.get("type") or "—")

    # ── SECTION 1: FLEET METRICS ────────────────────────────────────────────
    _section_header("Fleet Metrics")

    total = len(trucks)
    available = sum(
        1
        for t in trucks
        if str(t.get("status", "")).lower() == "available" or bool(t.get("available", False))
    )
    busy = total - available
    pct = 0 if total == 0 else int(round(100 * available / total))
    busy_pct = 0 if total == 0 else int(round(100 * busy / total))

    type_colors = {
        "Flatbed": "#1DE9B6",
        "Refrigerated": "#4FC3F7",
        "Container": "#B388FF",
        "Tanker": "#FFCA28",
        "Mini Van": "#00C851",
    }
    order = ["Flatbed", "Refrigerated", "Container", "Tanker", "Mini Van"]
    buckets: dict[str, dict[str, int]] = {}
    for t in trucks:
        typ = t.get("type") or "Flatbed"
        typ = str(typ)
        status = str(t.get("status", "available")).lower()
        is_avail = status == "available" or bool(t.get("available", False))
        b = buckets.setdefault(typ, {"total": 0, "available": 0})
        b["total"] += 1
        b["available"] += 1 if is_avail else 0

    st.markdown(
        _fleet_metrics_html(
            total=total,
            available=available,
            busy=busy,
            pct=pct,
            busy_pct=busy_pct,
            buckets=buckets,
            order=order,
            type_colors=type_colors,
        ),
        unsafe_allow_html=True,
    )

    st.divider()

    # ── SECTION 2: SHIPMENT MATCHER ─────────────────────────────────────────
    _section_header("Shipment Matcher")
    left, right = st.columns([1, 1.6], gap="large")

    with left:
        st.markdown("**Shipment input**  \n<span class='muted'>Enter the shipment details and request the best match.</span>", unsafe_allow_html=True)
        with st.form("shipment_form", clear_on_submit=False):
            weight_kg = st.number_input("Weight (kg)", min_value=0.0, value=1200.0, step=50.0)
            origin_city = st.text_input("Origin city", value="Mumbai")
            destination_city = st.text_input("Destination city", value="Pune")
            hour_of_day = st.slider("Hour of day", min_value=0, max_value=23, value=14)
            submitted = st.form_submit_button("Find Best Truck", type="primary", use_container_width=True)

        if submitted:
            try:
                match = _post_match(
                    {
                        "weight_kg": float(weight_kg),
                        "origin_city": origin_city,
                        "destination_city": destination_city,
                        "hour_of_day": int(hour_of_day),
                    }
                )
                st.session_state["last_match"] = match
                st.session_state["shipment_weight_kg"] = float(weight_kg)
                st.session_state.pop("selected_assignment", None)
                st.success("Top matches generated.")
            except Exception as e:
                st.error(f"Could not create assignment: {e}")

    with right:
        st.markdown("**Top recommendations**  \n<span class='muted'>Select one to finalize and generate the Claude explanation.</span>", unsafe_allow_html=True)

        match = st.session_state.get("last_match")
        if not match:
            st.info("No shipment submitted yet. Use the form on the left.")
        else:
            shipment_weight = float(st.session_state.get("shipment_weight_kg", match.get("weight_kg", 1200.0)) or 1200.0)
            recs = match.get("recommendations", [])
            delivery_id = match.get("delivery_id")

            for i, r in enumerate(recs[:3]):
                truck_id = r.get("truck_id")
                driver_name = r.get("driver_name") or "—"
                perf = float(r.get("driver_perf_score", 0.0))
                cap = float(r.get("capacity_kg", 0.0))
                pct_u = 0.0 if cap <= 0 else min(1.0, float(shipment_weight) / cap)
                try:
                    tid_int = int(truck_id) if truck_id is not None else None
                except (TypeError, ValueError):
                    tid_int = None
                t_label = truck_type_by_id.get(tid_int, "—") if tid_int is not None else "—"
                t_label_esc = html_lib.escape(str(t_label))

                with st.container(border=True):
                    st.markdown(
                        f'<div><strong>Truck `{html_lib.escape(str(truck_id))}`</strong>'
                        f'<span class="type-pill-rec">{t_label_esc}</span><br/>'
                        f"<span class='muted'>{html_lib.escape(str(driver_name))}</span></div>",
                        unsafe_allow_html=True,
                    )
                    top_row = st.columns([1.0, 1.0, 1.0, 1.2], gap="small")
                    top_row[0].metric("Perf", f"{perf:.2f}")
                    top_row[1].metric("Delay", f"{float(r.get('predicted_delay_min', 0)):.1f}m")
                    top_row[2].metric("ETA", f"{float(r.get('predicted_eta_min', 0)):.0f}m")
                    fuel_l = float(r.get("predicted_fuel_l", 0.0))
                    top_row[3].metric("Fuel / Cost", f"{fuel_l:.1f}L", f"{fuel_l * FUEL_PRICE_PER_L:,.0f}")

                    st.caption(f"Capacity {cap:.0f} kg · Shipment {shipment_weight:.0f} kg")
                    st.progress(pct_u, text=f"{shipment_weight:.0f} / {cap:.0f} kg")

                    if st.button(
                        f"SELECT THIS TRUCK (#{i+1})",
                        key=f"select_truck_{truck_id}",
                        type="primary" if i == 0 else "secondary",
                        use_container_width=True,
                    ):
                        try:
                            selected = _post_assign_select({"delivery_id": int(delivery_id), "truck_id": int(truck_id)})
                            st.session_state["selected_assignment"] = selected
                            st.success(selected.get("explanation") or "Selected. No explanation text.")
                        except Exception as e:
                            st.error(f"Could not finalize selection: {e}")

        sel = st.session_state.get("selected_assignment")
        if sel and sel.get("explanation"):
            st.success(sel["explanation"])

    st.divider()

    # ── SECTION 3: FLEET MAP ────────────────────────────────────────────────
    _section_header("Fleet Map")
    fm = _fleet_map(trucks)
    st_folium(fm, width=None, height=420, key="fleet_map", returned_objects=[])
    st.markdown(
        '<div class="fleet-legend"><span><span class="dot-g"></span> Available</span>'
        '<span><span class="dot-r"></span> Busy</span></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── SECTION 4: OPERATIONS HISTORY ───────────────────────────────────────
    _section_header("Operations History")
    tab_assign, tab_delivery, tab_drivers = st.tabs(["Past Assignments", "Delivery History", "Driver Performance"])

    with tab_assign:
        n_past = len(past_assignments)
        delays = [float(a["predicted_delay_min"]) for a in past_assignments if a.get("predicted_delay_min") is not None]
        fuels = [float(a["predicted_fuel_l"]) for a in past_assignments if a.get("predicted_fuel_l") is not None]
        avg_delay = mean(delays) if delays else None
        avg_fuel_tab = mean(fuels) if fuels else None
        d_part = f"{avg_delay:.1f} min" if avg_delay is not None else "—"
        f_part = f"{avg_fuel_tab:.1f} L" if avg_fuel_tab is not None else "—"
        st.caption(
            f"{n_past} assignments · Avg predicted delay: {d_part} · Avg fuel: {f_part}",
        )

        cols_pa = [
            "assignment_id",
            "truck_id",
            "delivery_id",
            "predicted_delay_min",
            "predicted_fuel_l",
            "predicted_eta_min",
            "created_at",
        ]
        if past_assignments:
            df_pa = pd.DataFrame(past_assignments)
            df_pa_view = df_pa[[c for c in cols_pa if c in df_pa.columns]].copy()
        else:
            df_pa_view = pd.DataFrame(columns=cols_pa)

        ev_pa = st.dataframe(
            df_pa_view,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key="past_assignments_table",
            hide_index=True,
        )
        rows_pa = _selection_rows_from_dataframe_event(ev_pa)
        if rows_pa:
            row_idx = int(rows_pa[0])
            if 0 <= row_idx < len(past_assignments):
                with st.expander("Full Claude explanation", expanded=True):
                    st.write(past_assignments[row_idx].get("explanation") or "(No explanation text.)")

    with tab_delivery:
        rated_deliveries = [a for a in past_assignments if a.get("delivery_on_time") is not None]
        if not rated_deliveries:
            st.info("No completed deliveries yet.")
        else:
            ok_ct = sum(1 for a in rated_deliveries if a.get("delivery_on_time") is True)
            on_pct = 100.0 * ok_ct / len(rated_deliveries)
            st.metric("On-time %", f"{on_pct:.1f}%", help=f"{ok_ct} on-time of {len(rated_deliveries)} rated")

            rows_html: list[str] = []
            for a in rated_deliveries:
                did = a.get("delivery_id", "")
                tid = a.get("truck_id", "")
                ot = a.get("delivery_on_time")
                eta = a.get("predicted_eta_min", "")
                cat = a.get("created_at", "")
                if ot is True:
                    ot_cell = '<span class="ok" style="font-size:1.1em;">✓</span>'
                else:
                    ot_cell = '<span class="bad" style="font-size:1.1em;">✗</span>'
                rows_html.append(
                    "<tr>"
                    f"<td>{html_lib.escape(str(did))}</td>"
                    f"<td>{html_lib.escape(str(tid))}</td>"
                    f"<td>{ot_cell}</td>"
                    f"<td>{html_lib.escape(str(eta))}</td>"
                    f"<td>{html_lib.escape(str(cat))}</td>"
                    "</tr>"
                )
            st.markdown(
                '<table class="ops-table"><thead><tr>'
                "<th>delivery_id</th><th>truck_id</th><th>delivery_on_time</th>"
                "<th>predicted_eta_min</th><th>created_at</th>"
                "</tr></thead><tbody>"
                + "".join(rows_html)
                + "</tbody></table>",
                unsafe_allow_html=True,
            )

    with tab_drivers:
        if not trucks:
            st.caption("No truck data.")
        else:
            df_drv = pd.DataFrame(trucks)
            chart_cols = [c for c in ("driver_name", "driver_perf_score") if c in df_drv.columns]
            if chart_cols == ["driver_name", "driver_perf_score"]:
                st.bar_chart(df_drv[["driver_name", "driver_perf_score"]], x="driver_name", y="driver_perf_score", height=320)
            cols_df = [c for c in ("driver_name", "truck_id", "type", "driver_perf_score", "status") if c in df_drv.columns]
            df_rank = df_drv[cols_df].copy() if cols_df else pd.DataFrame()
            if not df_rank.empty and "driver_perf_score" in df_rank.columns and "driver_name" in df_rank.columns:
                df_rank = df_rank.sort_values("driver_perf_score", ascending=False).reset_index(drop=True)
                names_out: list[str] = []
                for i, row in df_rank.iterrows():
                    nm = str(row.get("driver_name") or "—")
                    names_out.append(f"⭐ {nm}" if i == 0 else nm)
                df_rank["driver_name"] = names_out
            st.dataframe(df_rank, width="stretch", hide_index=True)

    st.divider()

    # ── SECTION 5: RECENT ASSIGNMENTS TABLE ────────────────────────────────
    _section_header("Recent Assignments")
    if not assignments:
        st.caption("No assignments yet. Submit a shipment above to create one.")
    else:
        df = pd.DataFrame(assignments)
        if "explanation" in df.columns:
            df["explanation_preview"] = df["explanation"].fillna("").astype(str).str.slice(0, 90) + "…"

        display_cols = [
            c
            for c in (
                "assignment_id",
                "delivery_id",
                "truck_id",
                "predicted_delay_min",
                "predicted_fuel_l",
                "predicted_eta_min",
                "explanation_preview",
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

        rows = _selection_rows_from_dataframe_event(event)
        if rows:
            row_idx = int(rows[0])
            if 0 <= row_idx < len(assignments):
                with st.expander("Full Claude explanation", expanded=True):
                    st.write(assignments[row_idx].get("explanation") or "(No explanation text.)")


def main() -> None:
    """Configure the Streamlit page and mount the auto-refreshing dashboard fragment."""
    st.set_page_config(page_title="AI Smart Load Matching", layout="wide")
    _inject_dark_css()
    st.title("AI Smart Load Matching")
    st.caption(f"API: `{API_BASE}` · Submit shipments and review recommendations.")

    _dashboard_fragment()


main()
