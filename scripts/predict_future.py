#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import joblib

HIST = "data/features_enriched.csv"
FUT  = "data/future_features_2025_2026.csv"
MODEL_PATH = "artifacts/model.joblib"
FEATURES_META = "artifacts/features.json"
OUT = "data/forecast_2025_2026.csv"

EVENT_COL = "event_intensity_new"

# Load model + feature list
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Missing artifacts/model.joblib. Run scripts/train.py first.")

model = joblib.load(MODEL_PATH)

if os.path.exists(FEATURES_META):
    with open(FEATURES_META, "r") as f:
        meta = json.load(f)
    features = meta.get("features", [])
else:
    # fallback: use common expected features
    features = [
        "days_to_dep","is_weekend_dep","season_month",
        "holiday_proximity", EVENT_COL,
        "origin_tavg","origin_prcp","origin_wspd",
        "dest_tavg","dest_prcp","dest_wspd",
    ]

# Load data
hist = pd.read_csv(HIST, parse_dates=["snapshot_date","dep_date"])
fut  = pd.read_csv(FUT, parse_dates=["snapshot_date","dep_date"])

# Normalize
hist["dep_month"] = pd.to_datetime(hist["dep_date"]).dt.month
fut["dep_month"]  = pd.to_datetime(fut["dep_date"]).dt.month

# Ensure event col
if EVENT_COL not in hist.columns:
    hist[EVENT_COL] = 0
if EVENT_COL not in fut.columns:
    fut[EVENT_COL] = 0

# ---- Learn event uplift ratios from last year ----
# ratio = median(event fares) / median(non-event fares)
# computed at route+airline+month level to avoid nonsense
grp_cols = ["origin","dest","airline","dep_month"]

def safe_ratio(event_med, nonevent_med):
    if nonevent_med is None or np.isnan(nonevent_med) or nonevent_med <= 0:
        return 1.0
    if event_med is None or np.isnan(event_med) or event_med <= 0:
        return 1.0
    r = float(event_med) / float(nonevent_med)
    # bound the uplift to keep realism (avoid extreme ratios from small samples)
    return float(np.clip(r, 0.85, 1.35))

uplifts = []
for key, g in hist.groupby(grp_cols):
    ev = g[g[EVENT_COL] > 0]["min_fare"]
    ne = g[g[EVENT_COL] == 0]["min_fare"]

    # require minimum data to trust uplift
    if len(ev) < 30 or len(ne) < 30:
        ratio = 1.0
    else:
        ratio = safe_ratio(ev.median(), ne.median())

    uplifts.append((*key, ratio, len(ev), len(ne)))

uplift_df = pd.DataFrame(uplifts, columns=grp_cols + ["event_uplift_ratio","n_event","n_nonevent"])

# Join future with uplift ratios
fut = fut.merge(uplift_df[grp_cols + ["event_uplift_ratio"]], on=grp_cols, how="left")
fut["event_uplift_ratio"] = fut["event_uplift_ratio"].fillna(1.0)

# ---- Model prediction ----
for c in features:
    if c not in fut.columns:
        fut[c] = 0.0

X = fut[features].copy().fillna(0)
fut["predicted_fare_base"] = model.predict(X)

# ---- Apply "same event pattern as last year" logic ----
# If future row is in an event window, we apply uplift learned from last year for that route/month/airline.
# If no event, ratio=1.0
fut["predicted_fare_adjusted"] = fut["predicted_fare_base"] * np.where(
    fut[EVENT_COL] > 0, fut["event_uplift_ratio"], 1.0
)

# Optional: make the adjustment visible as an explanation metric
fut["event_adjustment_pct"] = (fut["predicted_fare_adjusted"] - fut["predicted_fare_base"]) / np.maximum(fut["predicted_fare_base"], 1e-9) * 100.0

# Save
os.makedirs("data", exist_ok=True)
fut.to_csv(OUT, index=False)

print(f"[DONE] Wrote {OUT} rows={len(fut):,}")
print("Columns include: predicted_fare_base, predicted_fare_adjusted, event_uplift_ratio, event_adjustment_pct")
