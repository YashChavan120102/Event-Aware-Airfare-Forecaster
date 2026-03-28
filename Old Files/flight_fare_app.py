import streamlit as st
import pandas as pd
import altair as alt
from datetime import date, timedelta

# Import the core logic from your model file
from buywait import advise

# --- Configuration for Streamlit Page ---
st.set_page_config(
    page_title="Flight Fare Buy/Wait Advisor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Visual Styling (Dark/Sleek Theme) ---
st.markdown("""
    <style>
    /* 1. General Dark Theme Styling */
    .stApp {
        background-color: #1e1e2d; /* Dark background */
        color: #f0f2f6; /* Light text */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* 2. Custom Header */
    .header-title {
        background: linear-gradient(90deg, #6a85b6, #00C6FF); /* Sleek gradient */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5em;
        padding-bottom: 5px;
        margin-bottom: 20px;
        text-shadow: 0 4px 8px rgba(0, 198, 255, 0.2);
        display: flex;
        align-items: center;
    }
    .header-title svg {
        margin-right: 15px;
    }

    /* 3. The "Buy Now" / "Wait" Advice Card Styling */
    .advice-box {
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.5); /* Deep shadow for 3D effect */
    }
    .BUY_NOW {
        background-color: #2e7d32; /* Darker Green */
        border: 2px solid #66bb6a;
    }
    .WAIT {
        background-color: #ffb300; /* Darker Gold/Yellow */
        border: 2px solid #ffeb3b;
    }
    .advice-text {
        font-size: 2.8em;
        font-weight: 900;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: white; /* Ensure text is visible on colored background */
    }
    .reason-text {
        font-size: 1.2em;
        color: #e0e0e0;
        margin-top: 10px;
    }

    /* 4. Metric Styling (Cards) */
    [data-testid="stMetric"] {
        padding: 15px 10px;
        border: 1px solid #3d3d52;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        background-color: #28283c; /* Slightly lighter than app background */
    }
    [data-testid="stMetricLabel"] {
        color: #b0c4de; /* Light blue-gray for labels */
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8em;
        color: #00C6FF; /* Bright blue highlight */
        font-weight: 700;
    }

    /* 5. Selectbox/Input Styling (Make them dark) */
    .stSelectbox div, .stDateInput div {
        background-color: #28283c !important;
        border-color: #3d3d52 !important;
        color: #f0f2f6 !important;
    }
    .stButton>button {
        background-color: #00C6FF;
        color: #1e1e2d;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px #00a0cc;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #00a0cc;
        box-shadow: 0 2px #0084a3;
        transform: translateY(2px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Function for Plotting (Reusing from previous iterations) ---
@st.cache_data
def load_and_process_data(input_csv="data/features_enriched.csv", origin=None, dest=None, airline=None, dep_date=None):
    """Loads the data and computes the price curve for plotting."""
    try:
        df = pd.read_csv(input_csv, parse_dates=["snapshot_date", "dep_date"])
    except FileNotFoundError:
        st.error(f"Error: The file '{input_csv}' was not found. Please ensure it is in the same directory.")
        return None

    if dep_date is None:
        dep_date = df["dep_date"].max()

    sub = df[(df["origin"] == origin) & (df["dest"] == dest)]
    if airline:
        sub = sub[sub["airline"] == airline]
    
    if sub.empty:
        return None

    sub = sub[sub["dep_date"].dt.month == pd.to_datetime(dep_date).month]

    curve = sub.groupby("days_to_dep")["min_fare"].agg(["count", "mean", "median"]).reset_index()
    # Filter to only look at DTDs that have more than 5 historical points for reliability
    curve = curve[curve['count'] >= 5].sort_values("days_to_dep", ascending=False)
    return curve

# --- Mappings for UI ---
CITY_TO_IATA = {
    "Berlin, Germany": "BER",
    "London, UK": "LHR",
    "Mumbai, India": "BOM",
    "Sydney, Australia": "SYD"
}
IATA_TO_CITY = {v: k for k, v in CITY_TO_IATA.items()}

# --- Application UI Layout ---

# Custom Header with HTML and Icon
st.markdown('<p class="header-title"><svg viewBox="0 0 24 24" width="40" height="40" fill="none" stroke="#00C6FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-send"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>FLIGHT FARE AI ADVISOR</p>', unsafe_allow_html=True)
st.write("Secure the best price by leveraging AI-powered historical price curve analysis.")

# 1. Input Section
today = date.today()

with st.container():
    col1, col2, col3, col4 = st.columns([1.5, 1.5, 1, 1])

    with col1:
        origin_city = st.selectbox(
            "Departure From",
            options=sorted(CITY_TO_IATA.keys()),
            index=0,
            format_func=lambda x: x.split(',')[0]
        )
        origin_iata = CITY_TO_IATA[origin_city]

    with col2:
        dest_city = st.selectbox(
            "Destination To",
            options=sorted([c for c in CITY_TO_IATA.keys() if c != origin_city]),
            index=0,
            format_func=lambda x: x.split(',')[0]
        )
        dest_iata = CITY_TO_IATA[dest_city]

    with col3:
        max_date = today + timedelta(days=365) 
        
        dep_date = st.date_input(
            "Departure Date",
            min_value=today + timedelta(days=1),
            max_value=max_date,
            value=today + timedelta(days=90)
        )

    with col4:
        airline_options = ['All Airlines', 'BA', 'LH', 'AI', 'QF']
        selected_airline = st.selectbox(
            "Airline",
            options=airline_options,
            index=0
        )
        airline_param = selected_airline if selected_airline != 'All Airlines' else None

    # 2. Prediction Button
    st.markdown("---")
    if st.button("GET OPTIMAL BOOKING ADVICE", type="primary", use_container_width=True):
        # --- Prediction Logic ---
        try:
            # The buywait.py file is assumed to be the corrected version
            result = advise(
                input_csv="data/features_enriched.csv",
                origin=origin_iata,
                dest=dest_iata,
                airline=airline_param,
                dep_date=dep_date,
                today=today,
                min_savings_to_wait_pct=3.0
            )
            
            # --- Extract Results ---
            advice = result['advice']
            dtd_now = result['dtd_now']
            ideal_dtd = result['ideal_dtd']

            # --- 3. Display Results (Stylized Card) ---

            advice_text = f"BOOKING ADVICE: **{advice.replace('_', ' ')}**"
            reason_text = result['reason']
            
            st.markdown(f"""
                <div class="advice-box {advice}">
                    <p class="advice-text">{advice_text}</p>
                    <p class="reason-text">{reason_text}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Key Metrics
            st.subheader("Key Price Metrics")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric(label="Current Fare (USD)", value=f"${result['current_price']:.2f}")
            with col_m2:
                st.metric(label="Projected Min Fare (USD)", value=f"${result['projected_min_price']:.2f}")
            with col_m3:
                # Delta is the potential saving compared to current price
                st.metric(label="Potential Savings", 
                          value=f"{result['projected_savings_pct']:.2f}%", 
                          delta=f"{result['projected_savings_pct']:.2f}%" if result['projected_savings_pct'] > 0 else None
                )
            with col_m4:
                if ideal_dtd > 0:
                    ideal_date = dep_date - timedelta(days=ideal_dtd)
                    ideal_label = f"{ideal_date.strftime('%b %d')}"
                else:
                    ideal_label = "Today"
                
                st.metric(label="Ideal Booking Time", value=f"{ideal_dtd} days prior", delta=f"Target Date: {ideal_label}")

            st.markdown("---")

            # --- 4. Price Curve Visualization (Dark Mode friendly) ---
            st.subheader("Historical Price Analysis")

            plot_df = load_and_process_data(
                origin=origin_iata,
                dest=dest_iata,
                airline=airline_param,
                dep_date=dep_date
            )
            
            if plot_df is not None and not plot_df.empty:
                
                # Base chart with dark mode specific colors
                base = alt.Chart(plot_df).encode(
                    alt.X("days_to_dep", title="Days to Departure (DTD)", scale=alt.Scale(reverse=True, padding=5)),
                    alt.Y("mean", title="Average Fare (USD)"),
                    tooltip=["days_to_dep", alt.Tooltip("mean", format="$,.2f", title="Avg Fare"), "count"]
                ).properties(
                    height=400
                )
                
                # 1. Curve (Bright blue line)
                curve_chart = base.mark_line(color="#00C6FF").encode(
                    alt.OpacityValue(0.8)
                )

                # 2. Ideal Purchase Point (Target green dot)
                ideal_point = base.transform_filter(
                    alt.datum.days_to_dep == ideal_dtd
                ).mark_point(
                    filled=True,
                    size=200,
                    color="#66bb6a" # Bright green
                ).encode(
                    alt.Tooltip("mean", title="Projected Min Fare", format="$,.2f")
                )
                
                # 3. Current Days to Departure (Vertical line)
                current_line = alt.Chart(pd.DataFrame({'dtd': [dtd_now]})).mark_rule(color="#ffeb3b").encode( # Bright yellow
                    x='dtd',
                    size=alt.value(3),
                    tooltip=[alt.Tooltip("dtd", title="Days from Today")]
                )
                
                # Combine and display the chart
                final_chart = (curve_chart + ideal_point + current_line).properties(
                    title="Price vs. Days to Departure for Similar Flights"
                ).interactive()
                
                st.altair_chart(final_chart, use_container_width=True)

                st.caption(f"""
                <span style="color:#b0c4de;">The **cyan line** is the average historical fare. The **yellow line** marks today's DTD ($\mathbf{{T}}_{{DTD}} = {dtd_now}$ days). 
                The **green dot** is the optimal projected DTD ($\mathbf{{I}}_{{DTD}} = {ideal_dtd}$ days).</span>
                """, unsafe_allow_html=True)
            else:
                st.warning("No reliable historical data found (requires $\ge 5$ data points at each DTD) for this specific route and month combination. Try changing the month or selecting 'All Airlines'.")

        except ValueError as e:
            st.error(f"Prediction Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}. Check your 'buywait.py' for any new dependencies or changes.")