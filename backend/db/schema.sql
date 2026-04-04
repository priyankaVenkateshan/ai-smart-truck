-- Day 1 schema: trucks, drivers, deliveries, assignments

CREATE TABLE IF NOT EXISTS drivers (
  driver_id SERIAL PRIMARY KEY,
  driver_name TEXT NOT NULL,
  driver_perf_score DOUBLE PRECISION NOT NULL DEFAULT 0.75,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trucks (
  truck_id SERIAL PRIMARY KEY,
  driver_id INTEGER REFERENCES drivers(driver_id),
  capacity_kg DOUBLE PRECISION NOT NULL,
  fuel_eff_kmpl DOUBLE PRECISION NOT NULL,
  location_lat DOUBLE PRECISION NOT NULL,
  location_lng DOUBLE PRECISION NOT NULL,
  available BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS deliveries (
  delivery_id SERIAL PRIMARY KEY,
  origin_lat DOUBLE PRECISION NOT NULL,
  origin_lng DOUBLE PRECISION NOT NULL,
  dest_lat DOUBLE PRECISION NOT NULL,
  dest_lng DOUBLE PRECISION NOT NULL,
  distance_km DOUBLE PRECISION NOT NULL,
  weight_kg DOUBLE PRECISION NOT NULL,
  hour_of_day INTEGER NOT NULL CHECK (hour_of_day >= 0 AND hour_of_day <= 23),
  status TEXT NOT NULL DEFAULT 'pending',

  -- assigned/actuals (filled in later as the loop closes)
  assigned_truck_id INTEGER REFERENCES trucks(truck_id),
  actual_eta_min DOUBLE PRECISION,
  actual_delay_min DOUBLE PRECISION,
  actual_fuel_l DOUBLE PRECISION,
  on_time BOOLEAN,
  completed_at TIMESTAMPTZ,

  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS assignments (
  assignment_id SERIAL PRIMARY KEY,
  delivery_id INTEGER NOT NULL REFERENCES deliveries(delivery_id),
  truck_id INTEGER NOT NULL REFERENCES trucks(truck_id),

  predicted_eta_min DOUBLE PRECISION,
  predicted_delay_min DOUBLE PRECISION,
  predicted_fuel_l DOUBLE PRECISION,
  explanation TEXT,

  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trucks_available ON trucks(available);
CREATE INDEX IF NOT EXISTS idx_deliveries_status ON deliveries(status);
CREATE INDEX IF NOT EXISTS idx_assignments_created_at ON assignments(created_at DESC);

-- Saturday: auto-retrain checkpoint (single row id = 1)
CREATE TABLE IF NOT EXISTS retrain_state (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  last_retrain_at TIMESTAMPTZ
);

INSERT INTO retrain_state (id, last_retrain_at)
SELECT 1, NULL
WHERE NOT EXISTS (SELECT 1 FROM retrain_state WHERE id = 1);

