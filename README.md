# AI Smart Load Matching

This project is a small **end-to-end logistics demo**: synthetic trucks and deliveries live in **PostgreSQL**, a **greedy optimizer** assigns loads to available trucks, **XGBoost** models estimate delay, fuel, and ETA, **Claude** (optional) explains each assignment in plain English, and a **Streamlit** dashboard shows live truck positions and recent assignments. **MLflow** tracks training metrics, and completing deliveries can trigger **automatic retraining** after enough new outcomes.

## Prerequisites

- **Python 3.11+** (paths below use `py -3.11` on Windows; use `python3.11` on Linux/macOS if you prefer).
- **Docker Desktop** (for PostgreSQL + MLflow).

## Setup

### 1. Clone and enter the repo

```bash
git clone <your-repo-url>
cd "AI Smart Truck"
```

### 2. Environment variables

Copy `.env.example` to `.env` (e.g. `copy .env.example .env` in Windows **cmd**, or `Copy-Item .env.example .env` in **PowerShell**), then edit `.env`.

Important keys:

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | Use **`postgresql+psycopg://...`** for SQLAlchemy (`init_db`, `generate_data`). `backend/db/queries.py` (psycopg2) accepts the same URL and normalizes it automatically. |
| `MLFLOW_TRACKING_URI` | Usually `http://localhost:5000` |
| `ANTHROPIC_API_KEY` | Optional; without it, `/assign` still works but explanations are a placeholder |
| `API_BASE_URL` | Optional for the dashboard; default `http://localhost:8000` |

### 3. Install Python dependencies

```bash
py -3.11 -m pip install -r requirements.txt
```

### 4. Start Postgres and MLflow

```bash
docker compose up -d
```

Wait until Postgres is healthy (first run may take a minute).

### 5. Create database tables (first time or after schema changes)

```bash
py -3.11 -m backend.db.init_db
```

### 6. Load synthetic data (50 trucks, 500 deliveries)

```bash
py -3.11 -m data.generate.generate_data
```

### 7. Train models and log to MLflow

```bash
py -3.11 -m backend.training.train_models
```

Artifacts are written under `models/*.joblib`.

### 8. Run the API

```bash
py -3.11 -m uvicorn backend.api.main:app --reload
```

Open **http://localhost:8000/docs** for interactive API docs.

### 9. Run the dashboard (separate terminal)

```bash
py -3.11 -m streamlit run dashboard/app.py
```

---

## Full demo order (from scratch)

After `docker compose up -d`, run **once** to create/upgrade tables (required before loading data):

```bash
py -3.11 -m backend.db.init_db
```

Then the usual end-to-end sequence:

```bash
py -3.11 -m data.generate.generate_data
py -3.11 -m backend.training.train_models
py -3.11 -m uvicorn backend.api.main:app --reload
py -3.11 -m streamlit run dashboard/app.py
```

Use two terminals for the last two lines (API and dashboard together).

---

## API quick reference

### `POST /assign`

Assigns an available truck to a delivery, runs predictions, optional Claude explanation, saves an `assignments` row, and sets `assigned_truck_id` on the delivery.

Use **http://localhost:8000/docs** (Swagger UI) and **POST /assign**, or send JSON:

```json
{ "delivery_id": 1 }
```

### `POST /deliveries/{id}/complete`

Records actual outcomes, sets `on_time` and `completed_at`, updates the driver performance score, and may schedule auto-retrain in the background.

Example JSON body:

```json
{
  "actual_eta_min": 120,
  "actual_fuel_l": 45.0,
  "actual_delay_min": 8.0
}
```

Complete **after** you have assigned that delivery via `/assign` (so predictions / truck linkage exist).

### Other useful routes

- `GET /health` — liveness
- `GET /trucks/live` — all trucks + driver names and locations
- `GET /api/assignments/recent?limit=20` — latest assignments (used by the dashboard)

---

## MLflow UI

With Docker running, open **http://localhost:5000** in a browser. Open the **`ai_smart_truck`** experiment to see runs and **MAE** metrics for the three models.

---

## Tests

```bash
py -3.11 -m pytest tests/ -q
```

---

## Optional utilities

- Greedy optimizer smoke test (logs the routes dict): `py -3.11 -m backend.optimizer`
