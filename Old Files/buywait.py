import os
import json
import pandas as pd
import numpy as np
from datetime import date
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

EVENT_COL = "event_intensity_new"
EVENT_NAME_COL = "event_name"

def advise(input_csv="data/features_enriched.csv",
           origin="BER", dest="LHR", airline=None,
           dep_date=None, today=None,
           min_savings_to_wait_pct=3.0):
    """
    Predicts whether to 'BUY_NOW' or 'WAIT' using historical price curves.
    Also returns event info from event_intensity_new + event_name (if present).
    """

    # --- 1. Load and Prepare Data ---
    df = pd.read_csv(input_csv, parse_dates=["snapshot_date", "dep_date"])

    # Ensure columns exist (schema stability)
    if EVENT_COL not in df.columns:
        df[EVENT_COL] = 0
    if EVENT_NAME_COL not in df.columns:
        df[EVENT_NAME_COL] = ""

    # Defaults
    if dep_date is None:
        dep_date = df["dep_date"].max()
    if today is None:
        today = df["snapshot_date"].max()

    dep_date = pd.to_datetime(dep_date)
    today = pd.to_datetime(today)

    # --- 2. Filter Data ---
    sub = df[(df["origin"] == origin) & (df["dest"] == dest)].copy()
    if sub.empty:
        raise ValueError("No historical data found for the selected route.")

    if airline:
        sub = sub[sub["airline"] == airline].copy()
        if sub.empty:
            raise ValueError("No historical data found for the selected airline on this route.")

    # month-based pattern
    sub = sub[sub["dep_date"].dt.month == dep_date.month].copy()
    if sub.empty:
        raise ValueError("No historical data found for the selected month/route/airline combination.")

    dtd_now = int((dep_date - today).days)
    if dtd_now < 0:
        raise ValueError("Departure date must be in the future.")

    # --- 3. Current Price ---
    current = sub[(sub["dep_date"] == dep_date) & (sub["snapshot_date"] <= today)].sort_values("snapshot_date").tail(1)

    if current.empty:
        current_price = float(sub["min_fare"].median())
        if pd.isna(current_price):
            current_price = 9999.99
        event_level = int(sub[EVENT_COL].max()) if not sub.empty else 0
        event_names = ", ".join(sorted(set([x for x in sub[EVENT_NAME_COL].astype(str).tolist() if x.strip()])))
    else:
        current_price = float(current["min_fare"].iloc[0])
        event_level = int(current[EVENT_COL].iloc[0]) if pd.notna(current[EVENT_COL].iloc[0]) else 0
        event_names = str(current[EVENT_NAME_COL].iloc[0]) if pd.notna(current[EVENT_NAME_COL].iloc[0]) else ""

    # --- 4. Price Curve ---
    curve = sub.groupby("days_to_dep")["min_fare"].agg(["count", "mean", "median"]).reset_index().sort_values("days_to_dep", ascending=False)
    remain = curve[curve["days_to_dep"] <= dtd_now]
    if remain.empty:
        remain = curve

    ideal_dtd = int(remain.loc[remain["mean"].idxmin(), "days_to_dep"])
    min_mean = float(remain[remain["days_to_dep"] == ideal_dtd]["mean"].iloc[0])

    # --- 5. Advice ---
    projected_savings_pct = max(0.0, (current_price - min_mean) / max(current_price, 1e-9) * 100.0)

    if projected_savings_pct < min_savings_to_wait_pct:
        advice = "BUY_NOW"
        reason = f"Projected savings ({projected_savings_pct:.2f}%) are below the {min_savings_to_wait_pct}% threshold."
        ideal_booking_date = today.date()
        ideal_dtd_out = dtd_now
    else:
        advice = "WAIT"
        reason = f"Historical data suggests a future drop. Projected savings: {projected_savings_pct:.2f}%."
        ideal_booking_date = (dep_date - pd.Timedelta(days=ideal_dtd)).date()
        ideal_dtd_out = ideal_dtd

    return {
        "advice": advice,
        "reason": reason,
        "route": f"{origin}-{dest}" + (f" ({airline})" if airline else ""),
        "departure_date": str(dep_date.date()),
        "today": str(today.date()),
        "dtd_now": dtd_now,
        "current_price": round(current_price, 2),
        "projected_min_price": round(min_mean, 2),
        "projected_savings_pct": round(projected_savings_pct, 2),
        "ideal_dtd": ideal_dtd_out,
        "ideal_booking_date": str(ideal_booking_date),
        "event_intensity_new": int(event_level),
        "event_name": event_names
    }

if __name__ == "__main__":
    out = advise(
        origin="BER",
        dest="LHR",
        airline="BA",
        dep_date="2025-06-15",
        today="2025-04-01",
        min_savings_to_wait_pct=3.0
    )
    print(json.dumps(out, indent=2))
