"""
Tuesday: Train 3 XGBoost models and log to MLflow.
  - delay_model.joblib   → predicts actual_delay_min
  - fuel_model.joblib    → predicts actual_fuel_l
  - eta_model.joblib     → predicts actual_eta_min

Run with:
    py -3.11 -m backend.training.train_models
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

log = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── feature / target definitions ─────────────────────────────────────────────
DELAY_FEATURES = ["distance_km", "hour_of_day", "weight_kg", "driver_perf_score"]
FUEL_FEATURES  = ["distance_km", "weight_kg", "fuel_eff_kmpl"]
ETA_FEATURES   = ["distance_km", "driver_perf_score"]   # actual_delay added after synthesis


def _synthesise_training_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic training rows that mimic what the real DB will contain
    once deliveries are completed.  Uses the same distributions as
    data/generate/generate_data.py so the models are consistent.
    """
    rng = np.random.default_rng(seed)

    distance_km       = rng.uniform(50, 2500, n)
    hour_of_day       = rng.integers(0, 24, n)
    weight_kg         = np.clip(rng.normal(1200, 700, n), 1, 12_000)
    driver_perf_score = rng.uniform(0.5, 0.95, n)
    fuel_eff_kmpl     = rng.uniform(2.5, 6.5, n)

    # ── targets (realistic formulas + noise) ─────────────────────────────────
    # delay: longer distance, heavier load, night hours → more delay
    night_penalty     = np.where((hour_of_day < 6) | (hour_of_day > 21), 15, 0)
    actual_delay_min  = (
        distance_km * 0.05
        + weight_kg * 0.003
        + (1 - driver_perf_score) * 40
        + night_penalty
        + rng.normal(0, 5, n)
    ).clip(0)

    # fuel: distance / efficiency × load factor
    actual_fuel_l     = (
        distance_km / fuel_eff_kmpl
        * (1 + weight_kg / 20_000)
        + rng.normal(0, 2, n)
    ).clip(0.1)

    # eta: base speed 50 km/h + delay
    actual_eta_min    = (
        distance_km / 50 * 60
        + actual_delay_min
        + (1 - driver_perf_score) * 10
        + rng.normal(0, 3, n)
    ).clip(1)

    return pd.DataFrame({
        "distance_km":       distance_km,
        "hour_of_day":       hour_of_day.astype(float),
        "weight_kg":         weight_kg,
        "driver_perf_score": driver_perf_score,
        "fuel_eff_kmpl":     fuel_eff_kmpl,
        "actual_delay_min":  actual_delay_min,
        "actual_fuel_l":     actual_fuel_l,
        "actual_eta_min":    actual_eta_min,
    })


def _train_and_log(
    name: str,
    df: pd.DataFrame,
    features: list[str],
    target: str,
    params: dict,
) -> float:
    """Train one XGBRegressor, log to MLflow, save .joblib. Returns MAE."""
    X = df[features].dropna()
    y = df.loc[X.index, target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(**params, random_state=42, verbosity=0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)

    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_param("features", features)
        mlflow.log_param("target",   target)
        mlflow.log_metric("mae", round(mae, 4))
        mlflow.sklearn.log_model(model, artifact_path=name)

    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    log.info("trained %-20s MAE=%.3f saved %s", name, mae, path)
    return mae


def main() -> None:
    """Synthesise data, train three regressors, log MAE to MLflow, save `.joblib` files."""
    load_dotenv()

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("ai_smart_truck")

    log.info("Synthesising training data …")
    df = _synthesise_training_data(n=2000)

    log.info("Training models …")
    xgb_params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}

    # Add actual_delay as a feature for ETA model
    eta_features = ETA_FEATURES + ["actual_delay_min"]

    _train_and_log("delay_model", df, DELAY_FEATURES, "actual_delay_min", xgb_params)
    _train_and_log("fuel_model",  df, FUEL_FEATURES,  "actual_fuel_l",    xgb_params)
    _train_and_log("eta_model",   df, eta_features,   "actual_eta_min",   xgb_params)

    log.info("Training finished — view MLflow at %s", mlflow_uri)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
