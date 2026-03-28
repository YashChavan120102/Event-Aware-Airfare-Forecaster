#!/usr/bin/env python3
# scripts/train.py

import os, json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

INP = "data/features_enriched.csv"
os.makedirs("artifacts", exist_ok=True)

df = pd.read_csv(INP, parse_dates=["snapshot_date","dep_date"])

EVENT_COL = "event_intensity_new"

features = [
    "days_to_dep","is_weekend_dep","season_month",
    "holiday_proximity", EVENT_COL,
    "origin_tavg","origin_prcp","origin_wspd",
    "dest_tavg","dest_prcp","dest_wspd",
]

# Ensure all feature columns exist
for c in features:
    if c not in df.columns:
        df[c] = 0.0

X = df[features].fillna(0)
y = df["min_fare"].astype(float).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7
)

# Prefer LightGBM, fallback to RandomForest if LightGBM fails
backend = "lightgbm"
try:
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(n_estimators=400, learning_rate=0.05, random_state=7)
except Exception as e:
    backend = "sklearn_rf"
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=400, random_state=7, n_jobs=-1)

model.fit(X_train, y_train)
pred = model.predict(X_test)
mae = float(mean_absolute_error(y_test, pred))

# Save artifacts
with open("artifacts/metrics.json","w") as f:
    json.dump({"MAE": mae, "backend": backend}, f, indent=2)

with open("artifacts/features.json","w") as f:
    json.dump({"features": features, "event_column": EVENT_COL}, f, indent=2)

joblib.dump(model, "artifacts/model.joblib")

# Feature importance (if supported)
try:
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        fi = pd.DataFrame({"feature": features, "importance": importances})
        fi.sort_values("importance", ascending=False).to_csv(
            "artifacts/feature_importance.csv", index=False
        )
except Exception:
    pass

print("Training complete.")
print("Backend:", backend)
print("MAE:", round(mae, 2))
print("Saved: artifacts/model.joblib, artifacts/metrics.json, artifacts/features.json")
