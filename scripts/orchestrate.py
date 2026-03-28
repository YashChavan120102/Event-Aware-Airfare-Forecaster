#!/usr/bin/env python3
# scripts/orchestrate.py

"""
Stable orchestrator for EAAF.

- Reads data/fares.csv
- Builds holidays using python-holidays
- Fetches events from Ticketmaster (best-effort, non-breaking)
- Fetches daily weather from Meteostat
- Writes data/features_enriched.csv
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from meteostat import Point, Daily

# -------------------------
# Load environment
# -------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

TM_KEY = os.getenv("TICKETMASTER_KEY")

DATA_DIR = "data"
IN_FILE = os.path.join(DATA_DIR, "fares.csv")
OUT_FILE = os.path.join(DATA_DIR, "features_enriched.csv")

AIRPORTS = {
    "BER": {"city":"Berlin","country":"DE","lat":52.52,"lon":13.405,"elev":34},
    "LHR": {"city":"London","country":"GB","lat":51.47,"lon":-0.4543,"elev":25},
    "BOM": {"city":"Mumbai","country":"IN","lat":19.0896,"lon":72.8656,"elev":11},
    "SYD": {"city":"Sydney","country":"AU","lat":-33.8688,"lon":151.2093,"elev":58},
}

# -------------------------
# 1) Load fares
# -------------------------
print("[INFO] Loading fares ...")
fares = pd.read_csv(IN_FILE, parse_dates=["snapshot_date","dep_date"])
fares["dep_date"] = fares["dep_date"].dt.date

dep_min = pd.to_datetime(fares["dep_date"].min())
dep_max = pd.to_datetime(fares["dep_date"].max())

print(f"[INFO] Loaded {len(fares):,} fares")

# -------------------------
# 2) Holidays
# -------------------------
print("[INFO] Building holidays ...")
import holidays

rows = []
for code, cls in {
    "DE": holidays.Germany,
    "GB": holidays.UnitedKingdom,
    "IN": holidays.India,
    "AU": holidays.Australia
}.items():
    for year in range(dep_min.year, dep_max.year + 1):
        for d, name in cls(years=year).items():
            rows.append([code, pd.to_datetime(d).date(), name])

holidays_df = pd.DataFrame(rows, columns=["country","date","holiday"])
print(
    f"[INFO] Holidays fetched: {len(holidays_df):,} rows "
    f"for countries {sorted(holidays_df['country'].unique().tolist())}"
)

def holiday_proximity(dep_date, c1, c2):
    subset = holidays_df[holidays_df["country"].isin([c1, c2])]
    if subset.empty:
        return 999
    return int(
        np.abs(
            (pd.to_datetime(subset["date"]) - pd.to_datetime(dep_date))
            .dt.days
        ).min()
    )

# -------------------------
# 3) Ticketmaster (SAFE VERSION)
# -------------------------
def fetch_ticketmaster_events(cities, start_date, end_date):
    """
    Best-effort Ticketmaster fetch.
    Returns empty DataFrame if no events are available.
    """
    if not TM_KEY:
        print("[WARN] No Ticketmaster key. Skipping events.")
        return pd.DataFrame(columns=["city","start_date","end_date"])

    base = "https://app.ticketmaster.com/discovery/v2/events.json"
    rows = []

    for city in cities:
        params = {
            "apikey": TM_KEY,
            "city": city,
            "startDateTime": f"{start_date.date()}T00:00:00Z",
            "endDateTime": f"{end_date.date()}T23:59:59Z",
            "size": 100
        }

        r = requests.get(base, params=params, timeout=15)

        if r.status_code != 200:
            print(f"[WARN] Ticketmaster failed for {city}")
            continue

        events = r.json().get("_embedded", {}).get("events", [])
        for e in events:
            sd = e["dates"]["start"].get("localDate")
            if sd:
                rows.append([city, pd.to_datetime(sd).date(), pd.to_datetime(sd).date()])

        print(f"[INFO] Ticketmaster fetched {len(events)} events for {city}")

    return pd.DataFrame(rows, columns=["city","start_date","end_date"])

cities = [v["city"] for v in AIRPORTS.values()]
events_df = fetch_ticketmaster_events(cities, dep_min, dep_max)

# -------------------------
# 4) Weather (Meteostat)
# -------------------------
print("[INFO] Fetching weather ...")
weather_rows = []

for ap, meta in AIRPORTS.items():
    try:
        dfw = Daily(
            Point(meta["lat"], meta["lon"], meta["elev"]),
            dep_min,
            dep_max
        ).fetch().reset_index()
    except Exception as e:
        print(f"[WARN] Weather failed for {ap}: {e}")
        continue

    if dfw.empty:
        continue

    dfw["airport"] = ap
    dfw["date"] = dfw["time"].dt.date
    weather_rows.append(dfw)

weather_df = pd.concat(weather_rows, ignore_index=True)
if weather_df.empty:
    print("[WARN] No weather data fetched")
else:
    print(f"[INFO] Weather fetched: {len(weather_df):,} rows total")
    for ap in weather_df["airport"].unique():
        cnt = weather_df[weather_df["airport"] == ap].shape[0]
        print(f"       - {ap}: {cnt} daily rows")
        
# -------------------------
# 5) Enrich fares
# -------------------------
print("[INFO] Enriching fares ...")

fares["origin_city"] = fares["origin"].map(lambda x: AIRPORTS[x]["city"])
fares["dest_city"] = fares["dest"].map(lambda x: AIRPORTS[x]["city"])
fares["origin_ctry"] = fares["origin"].map(lambda x: AIRPORTS[x]["country"])
fares["dest_ctry"] = fares["dest"].map(lambda x: AIRPORTS[x]["country"])

fares["holiday_proximity"] = fares.apply(
    lambda r: holiday_proximity(r["dep_date"], r["origin_ctry"], r["dest_ctry"]),
    axis=1
)

def event_intensity(dep_date, o, d):
    if events_df.empty:
        return 0
    oc = AIRPORTS[o]["city"]
    dc = AIRPORTS[d]["city"]
    mask = (
        (events_df["city"].isin([oc, dc])) &
        (events_df["start_date"] <= dep_date) &
        (events_df["end_date"] >= dep_date)
    )
    return min(3, int(mask.sum()))

fares["event_intensity"] = fares.apply(
    lambda r: event_intensity(r["dep_date"], r["origin"], r["dest"]),
    axis=1
)

# Merge weather
ow = weather_df.rename(columns=lambda c: f"origin_{c}" if c not in ["airport","date"] else c)
dw = weather_df.rename(columns=lambda c: f"dest_{c}" if c not in ["airport","date"] else c)

fares = fares.merge(
    ow, how="left",
    left_on=["origin","dep_date"],
    right_on=["airport","date"]
)

fares = fares.merge(
    dw, how="left",
    left_on=["dest","dep_date"],
    right_on=["airport","date"]
)

fares["is_weekend_dep"] = fares["dep_date"].apply(
    lambda d: int(pd.to_datetime(d).weekday() >= 5)
)
fares["season_month"] = fares["dep_date"].apply(
    lambda d: pd.to_datetime(d).month
)

os.makedirs(DATA_DIR, exist_ok=True)
fares.to_csv(OUT_FILE, index=False)

print(f"[DONE] Saved enriched data → {OUT_FILE}")
