import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="EAAF Forecast", layout="wide")

HIST_PATH = "data/features_enriched.csv"
FCST_PATH = "data/forecast_2025_2026.csv"

EVENT_COL = "event_intensity_new"

_cache = getattr(st, "cache_data", None) or st.cache

@_cache
def load_hist():
    df = pd.read_csv(HIST_PATH, parse_dates=["snapshot_date","dep_date"])
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    df["dep_date"] = pd.to_datetime(df["dep_date"]).dt.date
    if EVENT_COL not in df.columns:
        df[EVENT_COL] = 0
    return df

@_cache
def load_fcst():
    df = pd.read_csv(FCST_PATH, parse_dates=["snapshot_date","dep_date"])
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    df["dep_date"] = pd.to_datetime(df["dep_date"]).dt.date
    if EVENT_COL not in df.columns:
        df[EVENT_COL] = 0
    return df

if not os.path.exists(HIST_PATH):
    st.error("Missing data/features_enriched.csv")
    st.stop()

if not os.path.exists(FCST_PATH):
    st.error("Missing data/forecast_2025_2026.csv. Run scripts/make_future_features.py then scripts/predict_future.py.")
    st.stop()

hist = load_hist()
fcst = load_fcst()

st.title("Event-Aware Airfare Forecaster — Next-Year Forecast")
st.caption("Forecasts next-year fares and explains the impact using last-year event-driven patterns (no event names shown).")

# ---- selectors ----
routes = sorted(fcst[["origin","dest"]].drop_duplicates().apply(lambda r: f"{r['origin']} → {r['dest']}", axis=1).tolist())
route = st.selectbox("Route", routes)
origin, dest = route.split(" → ")

airlines = sorted(fcst[(fcst["origin"]==origin) & (fcst["dest"]==dest)]["airline"].unique().tolist())
airline = st.selectbox("Airline", ["All"] + airlines)

sub_fcst = fcst[(fcst["origin"]==origin) & (fcst["dest"]==dest)].copy()
if airline != "All":
    sub_fcst = sub_fcst[sub_fcst["airline"] == airline].copy()

min_dep, max_dep = sub_fcst["dep_date"].min(), sub_fcst["dep_date"].max()
dep_date = st.date_input("Target Departure Date (Next Year)", value=max_dep, min_value=min_dep, max_value=max_dep)

sub_fcst = sub_fcst[sub_fcst["dep_date"] == dep_date].copy()
if sub_fcst.empty:
    st.warning("No forecast rows for selected filters.")
    st.stop()

# Choose “today” for booking decision simulation
min_today = sub_fcst["snapshot_date"].min()
max_today = sub_fcst["snapshot_date"].max()
today = st.date_input("Assumed booking date (today)", value=max_today, min_value=min_today, max_value=max_today)

# Use the row matching today snapshot (closest)
sub_fcst["snap_diff"] = (pd.to_datetime(sub_fcst["snapshot_date"]) - pd.to_datetime(today)).abs()
row_today = sub_fcst.sort_values("snap_diff").iloc[0]

price_now = float(row_today["predicted_fare_adjusted"])
event_level = int(row_today[EVENT_COL])
event_active = "Yes" if event_level > 0 else "No"

# Find best predicted booking time (minimum predicted price over snapshots)
best_row = sub_fcst.loc[sub_fcst["predicted_fare_adjusted"].idxmin()]
best_price = float(best_row["predicted_fare_adjusted"])
best_book_date = best_row["snapshot_date"]

savings_pct = max(0.0, (price_now - best_price) / max(price_now, 1e-9) * 100.0)
advice = "WAIT" if savings_pct >= 3.0 else "BUY NOW"

# ---- top KPIs ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Recommendation", advice)
c2.metric("Predicted price (today)", f"${price_now:,.0f}")
c3.metric("Best predicted booking date", str(best_book_date))
c4.metric("Potential savings", f"{savings_pct:.2f}%")

# ---- event context (no names) ----
st.subheader("Demand Context (Events)")
cc1, cc2, cc3 = st.columns(3)
cc1.metric("Event window active?", event_active)
cc2.metric("Event demand level", str(event_level))
cc3.metric("Event adjustment", f"{float(row_today.get('event_adjustment_pct', 0.0)):.2f}%")

# ---- Forecast trend chart ----
st.subheader("Next-Year Forecast Trend (as departure approaches)")
sub_fcst_sorted = sub_fcst.sort_values("snapshot_date")
fig = px.line(sub_fcst_sorted, x="snapshot_date", y="predicted_fare_adjusted", markers=True)
fig.update_layout(yaxis_title="Predicted fare (adjusted)", xaxis_title="Booking date (snapshot)")
st.plotly_chart(fig, use_container_width=True)

# ---- Last-year explanation panel ----
st.subheader("Last-Year Explanation (same route & month)")

# Compare with last year same month
target_month = pd.to_datetime(dep_date).month
hist_sub = hist[(hist["origin"]==origin) & (hist["dest"]==dest)].copy()
if airline != "All":
    hist_sub = hist_sub[hist_sub["airline"] == airline].copy()

hist_sub = hist_sub[pd.to_datetime(hist_sub["dep_date"]).dt.month == target_month].copy()

if hist_sub.empty:
    st.info("No last-year history available for this route/month to explain event behavior.")
else:
    # Build last year curves: event vs non-event
    hist_sub["is_event"] = (hist_sub[EVENT_COL] > 0).astype(int)

    curve_all = hist_sub.groupby("days_to_dep")["min_fare"].mean().reset_index().sort_values("days_to_dep", ascending=False)
    curve_ev  = hist_sub[hist_sub["is_event"]==1].groupby("days_to_dep")["min_fare"].mean().reset_index()
    curve_ne  = hist_sub[hist_sub["is_event"]==0].groupby("days_to_dep")["min_fare"].mean().reset_index()

    curve_ev["curve_type"] = "Last year: event window"
    curve_ne["curve_type"] = "Last year: non-event"
    curve_all["curve_type"] = "Last year: overall"

    explain = pd.concat([curve_all, curve_ev, curve_ne], ignore_index=True)
    explain = explain.dropna().sort_values("days_to_dep", ascending=False)

    fig2 = px.line(explain, x="days_to_dep", y="min_fare", color="curve_type", markers=False)
    fig2.update_layout(
        xaxis_title="Days to departure (historical)",
        yaxis_title="Average fare (last year)"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # High-level explanation text (no event names)
    ev_med = float(hist_sub[hist_sub[EVENT_COL] > 0]["min_fare"].median()) if (hist_sub[EVENT_COL] > 0).any() else np.nan
    ne_med = float(hist_sub[hist_sub[EVENT_COL] == 0]["min_fare"].median()) if (hist_sub[EVENT_COL] == 0).any() else np.nan

    if np.isfinite(ev_med) and np.isfinite(ne_med) and ne_med > 0:
        uplift = (ev_med / ne_med - 1) * 100.0
        st.write(
            f"Last year, during event windows, median fares were approximately **{uplift:.1f}%** different compared to non-event periods for this route/month. "
            "This learned pattern is applied to next-year forecasts when an event window is active."
        )
    else:
        st.write(
            "Last year did not have enough event vs non-event samples for a strong uplift estimate. "
            "Next-year forecasts still include event intensity as a model input."
        )
