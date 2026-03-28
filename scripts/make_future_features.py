#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from datetime import date, timedelta

# Output
OUT = "data/future_features_2025_2026.csv"

# Forecast window (exactly what you requested)
FUTURE_START = date(2025, 7, 1)
FUTURE_END   = date(2026, 6, 30)

# Snapshot horizons (how many days before departure we simulate price checks)
SNAPSHOT_DAYS = [1, 3, 7, 14, 21, 30, 45, 60, 90]

# Load historical to discover valid routes/airlines + for route metadata
HIST_PATH = "data/features_enriched.csv"
df_hist = pd.read_csv(HIST_PATH, parse_dates=["snapshot_date","dep_date"])
df_hist["dep_date"] = pd.to_datetime(df_hist["dep_date"]).dt.date

# Identify realistic route-airline combos from historical data
combos = (
    df_hist[["origin","dest","airline"]]
    .drop_duplicates()
    .sort_values(["origin","dest","airline"])
    .reset_index(drop=True)
)

# ---------- Holidays (python-holidays) ----------
try:
    import holidays as hol
except Exception as e:
    raise RuntimeError("Missing dependency 'holidays'. Install with: pip install holidays") from e

# Country mapping from your airports (keep consistent with your project)
AIRPORTS = {
    "BER": {"country":"DE"},
    "LHR": {"country":"GB"},
    "BOM": {"country":"IN"},
    "SYD": {"country":"AU"},
}

country_map = {
    "DE": hol.Germany,
    "GB": hol.UnitedKingdom,
    "IN": hol.India,
    "AU": hol.Australia,
}

def build_holidays(start_d: date, end_d: date):
    rows = []
    for code, cls in country_map.items():
        for y in range(start_d.year, end_d.year + 1):
            for d, name in cls(years=y).items():
                dd = pd.to_datetime(d).date()
                if start_d <= dd <= end_d:
                    rows.append((code, dd, name))
    return pd.DataFrame(rows, columns=["country","date","holiday_name"]).drop_duplicates()

hol_df = build_holidays(FUTURE_START - timedelta(days=120), FUTURE_END)  # buffer for proximity

def holiday_proximity(dep_date, origin_ctry, dest_ctry):
    subset = hol_df[hol_df["country"].isin([origin_ctry, dest_ctry])]
    if subset.empty:
        return 999
    d = pd.to_datetime(dep_date)
    return int(np.abs((pd.to_datetime(subset["date"]) - d).dt.days).min())

# ---------- Known future events (recurring annual windows) ----------
# We store: city -> [(label, start_month_day, end_month_day, intensity)]
# Then we convert to actual dates per year.
EVENT_TEMPLATES = {
    "Berlin": [
        ("Oktoberfest window", (9, 21), (10, 6), 3),
        ("ITB Berlin window", (3, 5), (3, 7), 3),
        ("Berlinale window", (2, 15), (2, 25), 3),
        ("Christmas markets window", (12, 1), (12, 23), 2),
        ("Easter travel window", (4, 16), (4, 21), 2),
    ],
    "London": [
        ("Wimbledon window", (7, 1), (7, 14), 3),
        ("Notting Hill Carnival window", (8, 24), (8, 26), 2),
        ("Christmas & New Year window", (12, 20), (1, 5), 3),  # cross-year
        ("London Marathon window", (4, 27), (4, 28), 2),
        ("Chelsea Flower Show window", (5, 20), (5, 25), 2),
    ],
    "Mumbai": [
        ("Ganesh Chaturthi window", (9, 7), (9, 17), 3),
        ("Diwali window", (11, 1), (11, 5), 3),
        ("IPL opening window", (3, 22), (4, 7), 2),
        ("Eid window", (4, 10), (4, 12), 2),
        ("Christmas & New Year window", (12, 20), (1, 2), 2),  # cross-year
    ],
    "Sydney": [
        ("New Year window", (12, 28), (1, 2), 3),  # cross-year
        ("Australian Open spillover", (1, 14), (1, 28), 2),
        ("Sydney Mardi Gras window", (2, 15), (3, 2), 3),
        ("Easter long weekend window", (4, 18), (4, 22), 2),
    ],
}

CITY_BY_AIRPORT = {"BER":"Berlin","LHR":"London","BOM":"Mumbai","SYD":"Sydney"}
CTRY_BY_AIRPORT = {"BER":"DE","LHR":"GB","BOM":"IN","SYD":"AU"}

def expand_event_windows_for_year(city: str, year: int):
    """Return list of (label, start_date, end_date, intensity) for that year.
       Handles cross-year windows (e.g., Dec -> Jan).
    """
    out = []
    for label, (sm, sd), (em, ed), intensity in EVENT_TEMPLATES.get(city, []):
        start = date(year, sm, sd)
        # cross-year case
        if em < sm:
            end = date(year + 1, em, ed)
        else:
            end = date(year, em, ed)
        out.append((label, start, end, intensity))
    return out

# Precompute all event windows covering the future horizon
EVENT_WINDOWS = {}
for city in EVENT_TEMPLATES:
    windows = []
    for y in range(FUTURE_START.year - 1, FUTURE_END.year + 1):
        windows.extend(expand_event_windows_for_year(city, y))
    EVENT_WINDOWS[city] = windows

def event_intensity_for_row(dep_date: date, origin: str, dest: str):
    d = dep_date
    oc = CITY_BY_AIRPORT.get(origin)
    dc = CITY_BY_AIRPORT.get(dest)

    intensity = 0
    labels = []

    for city in [oc, dc]:
        for label, s, e, w in EVENT_WINDOWS.get(city, []):
            if s <= d <= e:
                intensity = max(intensity, w)
                labels.append(label)

    # We keep labels in file for “explanation mode”, but you can hide them in UI if desired
    labels = ", ".join(sorted(set(labels)))
    return intensity, labels

# ---------- Weather proxies ----------
# Since Meteostat may fail or future weather isn't “known”, we use "seasonal normals proxy":
# compute month-wise medians from historical for each airport for key weather columns, then apply to future.
WEATHER_COLS = ["origin_tavg","origin_prcp","origin_wspd","dest_tavg","dest_prcp","dest_wspd"]

def build_weather_normals(hist: pd.DataFrame):
    # derive month normal per airport and prefix role (origin/dest)
    normals = {}
    # origin normals: based on historical origin_* for that origin airport
    for ap in CITY_BY_AIRPORT.keys():
        sub = hist[hist["origin"] == ap].copy()
        if sub.empty:
            continue
        sub["m"] = pd.to_datetime(sub["dep_date"]).dt.month
        for m in range(1, 13):
            sm = sub[sub["m"] == m]
            if sm.empty:
                continue
            normals[(ap, m, "origin")] = {
                "origin_tavg": float(sm["origin_tavg"].median()) if "origin_tavg" in sm else 0.0,
                "origin_prcp": float(sm["origin_prcp"].median()) if "origin_prcp" in sm else 0.0,
                "origin_wspd": float(sm["origin_wspd"].median()) if "origin_wspd" in sm else 0.0,
            }
    for ap in CITY_BY_AIRPORT.keys():
        sub = hist[hist["dest"] == ap].copy()
        if sub.empty:
            continue
        sub["m"] = pd.to_datetime(sub["dep_date"]).dt.month
        for m in range(1, 13):
            sm = sub[sub["m"] == m]
            if sm.empty:
                continue
            normals[(ap, m, "dest")] = {
                "dest_tavg": float(sm["dest_tavg"].median()) if "dest_tavg" in sm else 0.0,
                "dest_prcp": float(sm["dest_prcp"].median()) if "dest_prcp" in sm else 0.0,
                "dest_wspd": float(sm["dest_wspd"].median()) if "dest_wspd" in sm else 0.0,
            }
    return normals

weather_normals = build_weather_normals(df_hist)

def weather_features(origin: str, dest: str, dep_date: date):
    m = dep_date.month
    out = {
        "origin_tavg": 0.0, "origin_prcp": 0.0, "origin_wspd": 0.0,
        "dest_tavg": 0.0, "dest_prcp": 0.0, "dest_wspd": 0.0,
    }
    out.update(weather_normals.get((origin, m, "origin"), {}))
    out.update(weather_normals.get((dest, m, "dest"), {}))
    return out

# ---------- Build future feature rows ----------
rows = []
dep = FUTURE_START
while dep <= FUTURE_END:
    for r in combos.itertuples(index=False):
        origin, dest, airline = r.origin, r.dest, r.airline

        for dtd in SNAPSHOT_DAYS:
            snap = dep - timedelta(days=dtd)
            # keep snapshot in a realistic range: we allow snapshots starting from FUTURE_START - 120d
            # (it represents "we observed prices before dep_date" in simulation)
            if snap < FUTURE_START - timedelta(days=120):
                continue

            is_weekend = 1 if pd.to_datetime(dep).weekday() >= 5 else 0
            season_month = dep.month

            origin_ctry = CTRY_BY_AIRPORT.get(origin, "")
            dest_ctry = CTRY_BY_AIRPORT.get(dest, "")
            hol_prox = holiday_proximity(dep, origin_ctry, dest_ctry)

            e_int, e_label = event_intensity_for_row(dep, origin, dest)
            w = weather_features(origin, dest, dep)

            rows.append({
                "snapshot_date": snap,
                "origin": origin,
                "dest": dest,
                "airline": airline,
                "dep_date": dep,
                "days_to_dep": dtd,
                "is_weekend_dep": is_weekend,
                "season_month": season_month,
                "holiday_proximity": hol_prox,
                "event_intensity_new": e_int,
                "event_window_label": e_label,
                **w
            })

    dep += timedelta(days=1)

future = pd.DataFrame(rows)
future["snapshot_date"] = pd.to_datetime(future["snapshot_date"]).dt.date
future["dep_date"] = pd.to_datetime(future["dep_date"]).dt.date

os.makedirs("data", exist_ok=True)
future.to_csv(OUT, index=False)
print(f"[DONE] Wrote {OUT} rows={len(future):,} cols={len(future.columns)}")
