import os, json
import pandas as pd
import numpy as np

import dash
from dash import html, dcc, Input, Output, State, no_update
import plotly.express as px
import dash_bootstrap_components as dbc

# -----------------------------
# CONFIG
# -----------------------------
EVENT_COL = "event_intensity_new"
SAVINGS_THRESHOLD_PCT = 3.0

HIST_PATH = "data/features_enriched.csv"
FCST_PATH = "data/forecast_2025_2026.csv"
LAYOUT_PATH = "ui/layout.json"


# -----------------------------
# Helpers
# -----------------------------
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["snapshot_date", "dep_date"])
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    df["dep_date"] = pd.to_datetime(df["dep_date"]).dt.date
    if EVENT_COL not in df.columns:
        df[EVENT_COL] = 0
    return df


def fmt_money(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"${x:,.0f}"


def metric_card(id_, label, value="—", icon="bi bi-circle-fill"):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    html.I(className=icon, style={"fontSize": "1.2rem"}),
                    className="kpi-icon",
                ),
                html.Div(
                    [
                        html.Div(label, className="kpi-label"),
                        html.Div(value, id=id_, className="kpi-value"),
                    ],
                    style={"minWidth": 0},
                ),
            ],
            className="kpi-body",
        ),
        className="shadow-sm kpi-card",
    )


# -----------------------------
# Guardrails & load
# -----------------------------
if not os.path.exists(HIST_PATH):
    raise FileNotFoundError("Missing data/features_enriched.csv")
if not os.path.exists(FCST_PATH):
    raise FileNotFoundError("Missing data/forecast_2025_2026.csv")
if not os.path.exists(LAYOUT_PATH):
    raise FileNotFoundError("Missing ui/layout.json")

with open(LAYOUT_PATH, "r", encoding="utf-8") as f:
    UI = json.load(f)

hist = load_csv(HIST_PATH)
fcst = load_csv(FCST_PATH)

fcst["dep_month"] = pd.to_datetime(fcst["dep_date"]).dt.month
hist["dep_month"] = pd.to_datetime(hist["dep_date"]).dt.month

ORIGINS = sorted(fcst["origin"].dropna().unique().tolist())

DEP_MIN = fcst["dep_date"].min()
DEP_MAX = fcst["dep_date"].max()
SNAP_MIN = fcst["snapshot_date"].min()
SNAP_MAX = fcst["snapshot_date"].max()


# -----------------------------
# Dash App
# -----------------------------
external_stylesheets = [dbc.themes.LUX, dbc.icons.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = UI.get("app_title", "EAAF")


def header_bar():
    # Optional logo: assets/logo.png
    logo_img = html.Img(
        src=app.get_asset_url("logo.png"),
        style={"height": "65px", "marginRight": "12px"},
    )

    return dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                logo_img,
                                html.Div(
                                    [
                                        html.Div(
                                            UI.get("app_title", "Event-Aware Airfare Forecaster"),
                                            style={"fontSize": "22px", "fontWeight": "800", "letterSpacing": "0.8px"},
                                        ),
                                        html.Div(
                                            UI.get(
                                                "subtitle",
                                                "Next-year forecasts with event-aware demand signals (no event names shown).",
                                            ),
                                            className="text-muted",
                                            style={"marginTop": "2px"},
                                        ),
                                    ]
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                        width=9,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Span(
                                    "Forecast Dashboard",
                                    className="badge bg-dark",
                                    style={"fontSize": "0.95rem", "padding": "10px 14px", "borderRadius": "999px"},
                                )
                            ],
                            style={"display": "flex", "justifyContent": "flex-end", "alignItems": "center", "height": "100%"},
                        ),
                        width=3,
                    ),
                ],
                className="g-2",
                align="center",
            )
        ),
        className="shadow-sm",
        style={"marginTop": "10px"},
    )


# -----------------------------
# Layout
# -----------------------------
app.layout = dbc.Container(
    fluid=True,
    children=[
        header_bar(),
        html.Div(style={"height": "14px"}),

        dbc.Row(
            [
                # Controls
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(
                                        [html.I(className="bi bi-search", style={"marginRight": "8px"}), html.Span("Search")],
                                        style={"fontWeight": "800", "letterSpacing": "1px", "marginBottom": "14px"},
                                    ),

                                    html.Label("Origin"),
                                    dcc.Dropdown(
                                        id="origin",
                                        options=[{"label": o, "value": o} for o in ORIGINS],
                                        value=ORIGINS[0] if ORIGINS else None,
                                        clearable=False,
                                    ),
                                    html.Div(style={"height": "10px"}),

                                    html.Label("Destination"),
                                    dcc.Dropdown(id="dest", options=[], value=None, clearable=False),
                                    html.Div(style={"height": "10px"}),

                                    html.Label("Airline"),
                                    dcc.Dropdown(
                                        id="airline",
                                        options=[{"label": "All", "value": "All"}],
                                        value="All",
                                        clearable=False,
                                    ),
                                    html.Div(style={"height": "10px"}),

                                    # ---- Improved date pickers (InputGroup + icons) ----
                                    html.Label("Departure date"),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText(html.I(className="bi bi-calendar3")),
                                            dcc.DatePickerSingle(
                                                id="dep_date",
                                                min_date_allowed=str(DEP_MIN),
                                                max_date_allowed=str(DEP_MAX),
                                                date=str(DEP_MAX),
                                                display_format="DD MMM YYYY",
                                                className="date-picker",
                                            ),
                                        ],
                                        className="mb-2",
                                    ),

                                    html.Label("Booking date (today)"),
                                    dbc.InputGroup(
                                        [
                                            dbc.InputGroupText(html.I(className="bi bi-calendar-check")),
                                            dcc.DatePickerSingle(
                                                id="today",
                                                min_date_allowed=str(SNAP_MIN),
                                                max_date_allowed=str(SNAP_MAX),
                                                date=str(SNAP_MAX),
                                                display_format="DD MMM YYYY",
                                                className="date-picker",
                                            ),
                                        ],
                                        className="mb-2",
                                    ),

                                    dbc.Button(
                                        [html.I(className="bi bi-lightning-charge", style={"marginRight": "8px"}), "Show Results"],
                                        id="btn_search",
                                        color="dark",
                                        className="w-100",
                                    ),
                                    html.Div(id="validation_msg", className="text-danger mt-2", style={"fontSize": "0.9rem"}),

                                    html.Hr(),
                                    html.Div(
                                        "We compare today's predicted price vs the best remaining future booking date. "
                                        "Predictions exist at key booking windows (snapshots), so your booking date is mapped "
                                        "to the nearest snapshot.",
                                        className="text-muted",
                                        style={"fontSize": "0.85rem", "lineHeight": "1.35rem"},
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                        )
                    ],
                    width=3,
                ),

                # Results
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(metric_card("advice", "Recommendation", icon="bi bi-compass"), md=3, sm=6, xs=12),
                                dbc.Col(metric_card("price_today", "Predicted Price (Today)", icon="bi bi-cash-stack"), md=3, sm=6, xs=12),
                                dbc.Col(metric_card("savings", "Potential Savings", icon="bi bi-graph-down-arrow"), md=3, sm=6, xs=12),
                                dbc.Col(metric_card("event_level", "Event Demand Level", icon="bi bi-activity"), md=3, sm=6, xs=12),
                            ],
                            className="g-3",
                        ),

                        dbc.Row(
                            [
                                dbc.Col(metric_card("next_best_date", "Next Best Booking Date", icon="bi bi-calendar2-check"), md=3, sm=6, xs=12),
                                dbc.Col(metric_card("global_best_date", "Global Best Booking Date", icon="bi bi-award"), md=3, sm=6, xs=12),
                            ],
                            className="g-3 mt-1",
                        ),

                        dbc.Row(
                            [
                                dbc.Col(html.Div(id="status_text", className="text-muted", style={"paddingTop": "10px"}), width=12)
                            ],
                            className="mt-1",
                        ),

                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    [html.I(className="bi bi-graph-up", style={"marginRight": "8px"}), "Next-Year Forecast Trend"],
                                                    className="card-title-row",
                                                ),
                                                dcc.Graph(id="forecast_trend", config={"displayModeBar": False}, style={"height": "380px"}),
                                            ]
                                        ),
                                        className="shadow-sm graph-card",
                                    ),
                                    width=6,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    [html.I(className="bi bi-clock-history", style={"marginRight": "8px"}), "Last-Year Pattern (Event vs Non-Event)"],
                                                    className="card-title-row",
                                                ),
                                                dcc.Graph(id="last_year_curve", config={"displayModeBar": False}, style={"height": "380px"}),
                                            ]
                                        ),
                                        className="shadow-sm graph-card",
                                    ),
                                    width=6,
                                ),
                            ],
                            className="mt-2 g-3",
                        ),
                    ],
                    width=9,
                ),
            ],
            className="g-3",
        ),
    ],
)


# -----------------------------
# Callbacks
# -----------------------------
@app.callback(
    Output("dest", "options"),
    Output("dest", "value"),
    Input("origin", "value"),
)
def update_destinations(origin):
    if not origin:
        return [], None
    dests = sorted(fcst[fcst["origin"] == origin]["dest"].dropna().unique().tolist())
    opts = [{"label": d, "value": d} for d in dests]
    return opts, (dests[0] if dests else None)


@app.callback(
    Output("airline", "options"),
    Output("airline", "value"),
    Input("origin", "value"),
    Input("dest", "value"),
)
def update_airlines(origin, dest):
    if not origin or not dest:
        return [{"label": "All", "value": "All"}], "All"
    sub = fcst[(fcst["origin"] == origin) & (fcst["dest"] == dest)]
    airlines = sorted(sub["airline"].dropna().unique().tolist())
    opts = [{"label": "All", "value": "All"}] + [{"label": a, "value": a} for a in airlines]
    return opts, "All"


@app.callback(
    Output("dep_date", "min_date_allowed"),
    Output("dep_date", "max_date_allowed"),
    Output("dep_date", "date"),
    Input("origin", "value"),
    Input("dest", "value"),
    Input("airline", "value"),
)
def update_dep_calendar_bounds(origin, dest, airline):
    if not origin or not dest:
        return str(DEP_MIN), str(DEP_MAX), str(DEP_MAX)

    sub = fcst[(fcst["origin"] == origin) & (fcst["dest"] == dest)]
    if airline and airline != "All":
        sub = sub[sub["airline"] == airline]

    if sub.empty:
        return str(DEP_MIN), str(DEP_MAX), str(DEP_MAX)

    mn = sub["dep_date"].min()
    mx = sub["dep_date"].max()
    return str(mn), str(mx), str(mx)


@app.callback(
    Output("today", "min_date_allowed"),
    Output("today", "max_date_allowed"),
    Output("today", "date"),
    Input("origin", "value"),
    Input("dest", "value"),
    Input("airline", "value"),
    Input("dep_date", "date"),
)
def update_today_calendar_bounds(origin, dest, airline, dep_date):
    if not origin or not dest or not dep_date:
        return str(SNAP_MIN), str(SNAP_MAX), str(SNAP_MAX)

    dep = pd.to_datetime(dep_date).date()
    sub = fcst[(fcst["origin"] == origin) & (fcst["dest"] == dest) & (fcst["dep_date"] == dep)]
    if airline and airline != "All":
        sub = sub[sub["airline"] == airline]

    if sub.empty:
        return str(SNAP_MIN), str(SNAP_MAX), str(SNAP_MAX)

    mn = sub["snapshot_date"].min()
    mx = sub["snapshot_date"].max()
    return str(mn), str(mx), str(mx)


@app.callback(
    Output("validation_msg", "children"),
    Output("advice", "children"),
    Output("price_today", "children"),
    Output("next_best_date", "children"),
    Output("global_best_date", "children"),
    Output("savings", "children"),
    Output("event_level", "children"),
    Output("status_text", "children"),
    Output("forecast_trend", "figure"),
    Output("last_year_curve", "figure"),
    Input("btn_search", "n_clicks"),
    State("origin", "value"),
    State("dest", "value"),
    State("airline", "value"),
    State("dep_date", "date"),
    State("today", "date"),
    prevent_initial_call=True,
)
def run_search(n_clicks, origin, dest, airline, dep_date, today):
    if not origin or not dest or not dep_date or not today:
        return ("Please select Origin, Destination, Departure Date and Booking Date.",) + (no_update,) * 9

    dep = pd.to_datetime(dep_date).date()
    chosen_booking = pd.to_datetime(today).date()

    sub = fcst[(fcst["origin"] == origin) & (fcst["dest"] == dest) & (fcst["dep_date"] == dep)].copy()
    if airline and airline != "All":
        sub = sub[sub["airline"] == airline].copy()

    if sub.empty:
        return (f"No forecast data available for {origin} → {dest} with airline={airline}.",) + (no_update,) * 9

    snap_min = sub["snapshot_date"].min()
    snap_max = sub["snapshot_date"].max()
    if chosen_booking < snap_min or chosen_booking > snap_max:
        msg = f"Booking date must be between {snap_min} and {snap_max} for the selected departure date."
        return (msg,) + (no_update,) * 9

    sub = sub.sort_values("snapshot_date").copy()
    sub["snap_diff"] = (pd.to_datetime(sub["snapshot_date"]) - pd.to_datetime(chosen_booking)).abs()
    row_now = sub.sort_values("snap_diff").iloc[0]
    used_snapshot = row_now["snapshot_date"]

    price_now = float(row_now["predicted_fare_adjusted"])
    event_level = int(row_now[EVENT_COL]) if EVENT_COL in sub.columns else 0

    global_best_row = sub.loc[sub["predicted_fare_adjusted"].idxmin()]
    global_best_date = global_best_row["snapshot_date"]

    future_sub = sub[sub["snapshot_date"] >= used_snapshot].copy()
    if future_sub.empty:
        future_sub = sub.copy()

    next_best_row = future_sub.loc[future_sub["predicted_fare_adjusted"].idxmin()]
    next_best_date = next_best_row["snapshot_date"]
    next_best_price = float(next_best_row["predicted_fare_adjusted"])

    savings_pct = max(0.0, (price_now - next_best_price) / max(price_now, 1e-9) * 100.0)
    advice = "WAIT" if (next_best_date != used_snapshot and savings_pct >= SAVINGS_THRESHOLD_PCT) else "BUY NOW"

    status = (
        f"Selected booking date: {chosen_booking} (mapped to snapshot: {used_snapshot}) | "
        f"Next best date: {next_best_date} | "
        f"Event-active: {'Yes' if event_level > 0 else 'No'} | Level: {event_level}"
    )

    fig1 = px.line(sub, x="snapshot_date", y="predicted_fare_adjusted", markers=True)
    fig1.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Predicted Fare (Adjusted)",
        xaxis_title="Booking Date (Snapshot)",
        showlegend=False,
    )
    fig1.add_vline(x=used_snapshot, line_width=2, line_dash="dash")
    fig1.add_vline(x=next_best_date, line_width=2, line_dash="dot")

    dep_month = int(pd.to_datetime(dep).month)
    hist_sub = hist[(hist["origin"] == origin) & (hist["dest"] == dest) & (hist["dep_month"] == dep_month)].copy()
    if airline and airline != "All":
        hist_sub = hist_sub[hist_sub["airline"] == airline].copy()

    if hist_sub.empty:
        fig2 = px.line(pd.DataFrame({"x": [0], "y": [0]}), x="x", y="y")
        fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_visible=False, yaxis_visible=False)
    else:
        hist_sub["curve"] = np.where(hist_sub[EVENT_COL] > 0, "event", "non_event")
        curve = hist_sub.groupby(["days_to_dep", "curve"])["min_fare"].mean().reset_index()
        fig2 = px.line(curve, x="days_to_dep", y="min_fare", color="curve")
        fig2.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Days to Departure (Last Year)",
            yaxis_title="Avg Fare (Last Year)",
        )

    return (
        "",
        advice,
        fmt_money(price_now),
        str(next_best_date),
        str(global_best_date),
        f"{savings_pct:.2f}%",
        str(event_level),
        status,
        fig1,
        fig2,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
