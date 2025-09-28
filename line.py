import streamlit as st
import pandas as pd
import numpy as np
import math
import numbers
import pickle
import csv
from io import BytesIO
from datetime import datetime, timedelta
from pathlib import Path

# Mapping / heatmap
import folium
from folium.plugins import HeatMap
from streamlit.components.v1 import html as st_html

# Optional extras (safe to skip if not installed)
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# Plotly for improved line charts
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# ----------------------------
# CONFIG & CONSTANTS
# ----------------------------
FAILED_LOG_CSV = "failed_churn_predictions_log.csv"
MODEL_FILE = "auto_taxi_churn_model.pkl"
ENC_FILE = "auto_taxi_encoders.pkl"
DATA_FILE = "auto_taxi_churn.csv"

CITY_COORDS = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Pune": (18.5204, 73.8567),
    "Bangalore": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714)
}

# Ensure log file exists
if not Path(FAILED_LOG_CSV).exists():
    with open(FAILED_LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "input_row", "stage", "error"])

# ----------------------------
# Utilities
# ----------------------------
def clamp(v, low, high):
    try:
        if v is None:
            return low
        return max(low, min(high, v))
    except Exception:
        return low

def is_finite_number(x):
    return isinstance(x, numbers.Number) and not (math.isinf(x) or math.isnan(x))

def safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return float(default)
        val = float(x)
        if math.isinf(val) or math.isnan(val):
            return float(default)
        return val
    except Exception:
        return float(default)

def log_failed_prediction(input_row: dict, stage: str, error_msg: str):
    ts = datetime.now().isoformat()
    with open(FAILED_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ts, repr(input_row), stage, error_msg])

# ----------------------------
# Heatmap helpers (reuse original look)
# ----------------------------
def generate_hotspots_from_churn(city, churn_prob, spread_km=12.0, seed=None):
    rng = np.random.default_rng(seed)
    center = CITY_COORDS.get(city, (20.0, 77.0))
    lat0, lon0 = center
    num_spots = clamp(int(churn_prob * 20) + 4, 3, 30)
    pts = []
    base_users = int(clamp(churn_prob * 200, 5, 500))
    for _ in range(num_spots):
        angle = rng.random() * 2 * math.pi
        r = (rng.random() ** 0.5) * spread_km
        dlat = (r / 111.0) * math.cos(angle)
        dlon = (r / (111.0 * math.cos(math.radians(lat0)))) * math.sin(angle)
        lat = lat0 + dlat
        lon = lon0 + dlon
        weight = max(1, int(base_users / (num_spots + rng.random() * 3) * (0.5 + churn_prob)))
        pts.append((lat, lon, weight))
    return pts

def create_heatmap_html_churn(city, hotspots, events=None, map_zoom=12, tiles="CartoDB dark_matter"):
    center = CITY_COORDS.get(city, (20.0, 77.0))
    fmap = folium.Map(location=center, tiles=tiles, zoom_start=map_zoom)
    heat_data = [[h[0], h[1], h[2]] for h in hotspots]
    HeatMap(heat_data, radius=18, blur=25, max_zoom=13).add_to(fmap)
    if events:
        for ev in events:
            if ev.get("lat") and ev.get("lon"):
                popup = folium.Popup(f"{ev.get('name')}<br/>{ev.get('time')}", max_width=250)
                folium.CircleMarker(location=(ev["lat"], ev["lon"]), radius=7, popup=popup,
                                    color="orange", fill=True, fill_color="orange").add_to(fmap)
    return fmap._repr_html_()

# ----------------------------
# Load model & encoders
# ----------------------------
def load_model_and_encoders():
    if not Path(MODEL_FILE).exists() or not Path(ENC_FILE).exists():
        raise FileNotFoundError(f"Missing model/encoders. Expected {MODEL_FILE} and {ENC_FILE} in project folder.")
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(ENC_FILE, "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

# ----------------------------
# Load dataset (static)
# ----------------------------
def load_dataset():
    p = Path(DATA_FILE)
    if p.exists():
        try:
            df = pd.read_csv(p)
            return df, DATA_FILE
        except Exception:
            pass
    cols = ["customer_id","day_of_week","time_of_day","location","weather","payment_method",
            "zone_type","gender","trip_type","avg_trips_per_week","avg_estimated_price",
            "avg_spent_per_week","last_trip_days_ago","tenure_days","churn","churn_prob"]
    return pd.DataFrame(columns=cols), f"{DATA_FILE} (not found)"

# ----------------------------
# Encoding helper
# ----------------------------
def safe_encode(encoders, col, value):
    enc = encoders.get(col)
    if enc is None:
        return value
    try:
        if value not in getattr(enc, "classes_", []):
            classes_lower = {str(c).lower(): c for c in enc.classes_}
            matched = classes_lower.get(str(value).lower())
            use_val = matched if matched is not None else enc.classes_[0]
            return enc.transform([use_val])[0]
        return enc.transform([value])[0]
    except Exception:
        try:
            return enc.transform([enc.classes_[0]])[0]
        except Exception:
            return 0

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Auto & Taxi Churn", layout="wide")

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #ffecd2, #fcb69f, #ffdde1, #ee9ca7);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(6px);
    border-right: 2px solid #ddd;
}
h1 { text-align: center; color: #2c3e50; font-family: 'Trebuchet MS', sans-serif; }
div.stButton > button {
    background-color: #ff914d; color: white; border-radius: 12px; padding: 10px 18px; font-weight: 700;
}
div.stButton > button:hover { background-color: #ff6b00; transform: scale(1.03); }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.image("https://cdn-icons-png.flaticon.com/512/3097/3097144.png", width=100)
st.title("🚖 Auto & Taxi — Churn Prediction")
st.markdown("### 🌆 Predict which customers are likely to **churn** — with heatmaps, charts & a modern UI")

# Load data & model
df, used_csv = load_dataset()
try:
    model, encoders = load_model_and_encoders()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar inputs
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/684/684908.png", width=90)
st.sidebar.header("Enter Customer Profile")

def safe_unique(col, default_list):
    if col in df.columns and not df[col].dropna().empty:
        return sorted(df[col].dropna().astype(str).unique().tolist())
    return default_list

day = st.sidebar.selectbox("📅 Day of Week", safe_unique("day_of_week",
                         ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]))
time_of_day = st.sidebar.selectbox("⏰ Time of Day", safe_unique("time_of_day",
                         ["Morning","Afternoon","Evening","Night"]))
location = st.sidebar.selectbox("🌆 City", safe_unique("location", list(CITY_COORDS.keys())))
weather = st.sidebar.selectbox("🌦 Weather", safe_unique("weather", ["Clear","Cloudy","Rainy","Foggy"]))
payment_method = st.sidebar.selectbox("💳 Payment Method", safe_unique("payment_method", ["Cash","Card","UPI"]))
zone_type = st.sidebar.selectbox("🏙 Zone Type", safe_unique("zone_type", ["Commercial","Residential","Mixed"]))
gender = st.sidebar.selectbox("🧍 Gender", safe_unique("gender", ["Male","Female","Other"]))
trip_type = st.sidebar.selectbox("🚘 Trip Type", safe_unique("trip_type", ["Auto","Taxi"]))

avg_trips_per_week = st.sidebar.number_input("📊 Avg trips per week", min_value=0.0, value=3.0, step=0.5)
avg_estimated_price = st.sidebar.number_input("💰 Avg estimated price per trip (₹)", min_value=0.0, value=80.0, step=1.0)
avg_spent_per_week = st.sidebar.number_input("💵 Avg spent per week (₹)", min_value=0.0,
                                             value=round(avg_trips_per_week * avg_estimated_price, 1), step=1.0)
last_trip_days_ago = st.sidebar.number_input("📅 Days since last trip", min_value=0, value=5, step=1)
tenure_days = st.sidebar.number_input("📆 Tenure (days)", min_value=1, value=180, step=1)

predict_clicked = st.sidebar.button("🔮 Predict Churn & Show Heatmap")

# ----------------------------
# Prediction & churn logic
# ----------------------------
if predict_clicked:
    raw_inputs = {
        "day_of_week": day,
        "time_of_day": time_of_day,
        "location": location,
        "weather": weather,
        "payment_method": payment_method,
        "zone_type": zone_type,
        "gender": gender,
        "trip_type": trip_type,
        "avg_trips_per_week": float(avg_trips_per_week),
        "avg_estimated_price": float(avg_estimated_price),
        "avg_spent_per_week": float(avg_spent_per_week),
        "last_trip_days_ago": int(last_trip_days_ago),
        "tenure_days": int(tenure_days),
    }

    try:
        input_row = {k: safe_encode(encoders, k, v) if k in encoders else v for k, v in raw_inputs.items()}
        input_df = pd.DataFrame([input_row])
        if hasattr(model, "feature_names_in_"):
            model_features = list(model.feature_names_in_)
            input_df = input_df.reindex(columns=model_features, fill_value=0)
        proba = float(model.predict_proba(input_df)[0][1])
        pred_label = int(model.predict(input_df)[0])
        if not is_finite_number(proba):
            raise ValueError(f"Non-finite probability: {proba}")
    except Exception as e:
        log_failed_prediction(raw_inputs, "model_predict", str(e))
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.success(f"📈 Predicted Churn Probability: **{proba:.2%}**")
    c1, c2, c3, c4 = st.columns([2,2,2,2])
    risk_text = "⚠️ HIGH RISK" if proba >= 0.5 else "✅ LOW RISK"
    c1.metric("Churn Risk", risk_text)
    c2.metric("Churn Probability", f"{proba:.2%}")
    c3.metric("Days since last trip", f"{raw_inputs['last_trip_days_ago']} days")
    c4.metric("Avg trips/week", f"{raw_inputs['avg_trips_per_week']}")

    if proba >= 0.5:
        st.warning("This customer shows a high probability of churning. Suggest targeted retention offers, messages, or discounts.")
    else:
        st.info("Customer appears engaged. Continue regular engagement strategies.")

    try:
        hotspots = generate_hotspots_from_churn(raw_inputs["location"], proba, spread_km=10.0, seed=42)
        map_html = create_heatmap_html_churn(raw_inputs["location"], hotspots, events=[], map_zoom=12)
        st.markdown("### 🔥 Churn Heatmap (weighted by churn probability)")
        st_html(map_html, height=480)
    except Exception as e:
        log_failed_prediction(raw_inputs, "heatmap", str(e))
        st.info("Could not render churn heatmap.")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append((
        ts,
        raw_inputs["day_of_week"],
        raw_inputs["location"],
        raw_inputs["trip_type"],
        round(proba, 4),
        "HIGH" if proba >= 0.5 else "LOW",
        raw_inputs["avg_trips_per_week"],
        raw_inputs["avg_spent_per_week"],
        raw_inputs["last_trip_days_ago"]
    ))

# ----------------------------
# Insights & charts (Static + Dynamic with Tabs)
# ----------------------------
st.markdown("## 📊 Insights & Trends")

tab1, tab2 = st.tabs(["📂 Static (Dataset)", "⚡ Dynamic (Predictions)"])

# --- Tab 1: Static from dataset ---
with tab1:
    st.markdown("### 📂 Insights from Dataset (CSV)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌆 Average Churn Rate by City")
        if "location" in df.columns and "churn" in df.columns and not df.empty:
            city_data = df.groupby("location")["churn"].mean().sort_values(ascending=False)
            st.bar_chart(city_data)
        else:
            st.info("Dataset missing 'location' or 'churn' for this chart.")
    with col2:
        st.markdown("### 🌦 Churn by Weather")
        if "weather" in df.columns and "churn" in df.columns and not df.empty:
            weather_data = df.groupby("weather")["churn"].mean().sort_values(ascending=False)
            st.bar_chart(weather_data)
        else:
            st.info("Dataset missing 'weather' or 'churn' for this chart.")
    st.markdown("### 🚘 Auto vs Taxi Churn Rate")
    if "trip_type" in df.columns and "churn" in df.columns and not df.empty:
        trip_type_data = df.groupby("trip_type")["churn"].mean().sort_values(ascending=False)
        st.bar_chart(trip_type_data)
    else:
        st.info("Dataset missing 'trip_type' or 'churn' for this chart.")
    st.markdown("### 📅 Churn vs Days Since Last Trip (distribution)")
    if "last_trip_days_ago" in df.columns and "churn" in df.columns and not df.empty:
        df_plot = df.copy()
        df_plot["days_bin"] = pd.cut(df_plot["last_trip_days_ago"].fillna(0), bins=[-1,1,7,14,30,60,180,10000],
                                     labels=["0","1-7","8-14","15-30","31-60","61-180","180+"])
        bin_data = df_plot.groupby("days_bin")["churn"].mean()
        st.bar_chart(bin_data)
    else:
        st.info("Dataset missing 'last_trip_days_ago' or 'churn' for this chart.")

# --- Tab 2: Dynamic from predictions (violin plots with jitter fix) ---
with tab2:
    st.markdown("### ⚡ Insights from Your Predictions (Dynamic)")
    if st.session_state.history:
        hist_cols = ["Timestamp","Day","City","Trip Type","ChurnProb","Risk",
                     "AvgTrips","AvgSpentPerWeek","DaysSinceLastTrip"]
        hist_df = pd.DataFrame(st.session_state.history, columns=hist_cols)

        # Ensure numeric & proper types
        hist_df["ChurnProb"] = pd.to_numeric(hist_df["ChurnProb"], errors="coerce").fillna(0.0)
        hist_df["AvgTrips"] = pd.to_numeric(hist_df["AvgTrips"], errors="coerce").fillna(0.0)
        hist_df["DaysSinceLastTrip"] = pd.to_numeric(hist_df["DaysSinceLastTrip"], errors="coerce").fillna(0.0)
        hist_df["Timestamp"] = pd.to_datetime(hist_df["Timestamp"])

        # Proper weekday order
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        hist_df["Day"] = pd.Categorical(hist_df["Day"], categories=day_order, ordered=True)

        # --- Add jitter if only 1 value per category (for violins to render correctly) ---
        hist_df_violin = hist_df.copy()
        for col in ["City", "Trip Type"]:
            counts = hist_df_violin[col].value_counts()
            for cat, cnt in counts.items():
                if cnt == 1:
                    idx = hist_df_violin[hist_df_violin[col] == cat].index[0]
                    val = hist_df_violin.at[idx, "ChurnProb"]
                    extra_rows = pd.DataFrame([
                        {**hist_df_violin.loc[idx].to_dict(), "ChurnProb": val * 0.98},
                        {**hist_df_violin.loc[idx].to_dict(), "ChurnProb": val * 1.02}
                    ])
                    hist_df_violin = pd.concat([hist_df_violin, extra_rows], ignore_index=True)

        col1, col2 = st.columns(2)

        # --- City-level violin plot ---
        with col1:
            st.markdown("### 🌆 Churn Probability Distribution by City")
            if PLOTLY_AVAILABLE and not hist_df_violin.empty:
                fig = px.violin(
                    hist_df_violin,
                    y="ChurnProb", x="City", color="City",
                    box=True, points="all",
                    hover_data=["Risk","AvgTrips","AvgSpentPerWeek","DaysSinceLastTrip"],
                    title="Churn Probability Distribution per City"
                )
                fig.update_layout(yaxis_title="Churn Probability (%)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data yet for City violin plot.")

        # --- Trip Type violin plot ---
        with col2:
            st.markdown("### 🚘 Churn Probability Distribution by Trip Type")
            if PLOTLY_AVAILABLE and not hist_df_violin.empty:
                fig = px.violin(
                    hist_df_violin,
                    y="ChurnProb", x="Trip Type", color="Trip Type",
                    box=True, points="all",
                    hover_data=["City","Risk","AvgTrips","DaysSinceLastTrip"],
                    title="Churn Probability Distribution per Trip Type"
                )
                fig.update_layout(yaxis_title="Churn Probability (%)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data yet for Trip Type violin plot.")

        # --- Day-of-week bar chart ---
        st.markdown("### 📅 Average Predicted Churn by Day of Week")
        day_avg = hist_df.groupby("Day")["ChurnProb"].mean().reset_index()
        if PLOTLY_AVAILABLE and not day_avg.empty:
            fig = px.bar(
                day_avg, x="Day", y="ChurnProb",
                text=day_avg["ChurnProb"].map(lambda x: f"{x:.2%}"),
                title="Average Churn Probability per Weekday"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis_title="Churn Probability (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data yet for Day-of-week chart.")

        # --- Heatmap ---
        st.markdown("### 🔥 Heatmap of Predicted Churns (from history)")
        heat_data = []
        for _, row in hist_df.iterrows():
            loc = row["City"]
            if loc in CITY_COORDS:
                lat, lon = CITY_COORDS[loc]
                heat_data.append([lat, lon, float(row["ChurnProb"]) * 10.0])

        if heat_data:
            fmap = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB dark_matter")
            HeatMap(heat_data, radius=18, blur=25, max_zoom=10).add_to(fmap)
            st_html(fmap._repr_html_(), height=420)
        else:
            st.info("No predictions yet to display on heatmap.")
    else:
        st.info("No predictions yet. Make one from the sidebar to see dynamic charts.")



# ----------------------------
# History + Export
# ----------------------------
st.markdown("## 🧾 Prediction History")
if st.session_state.history:
    hist_cols = ["Timestamp","Day","City","Trip Type","ChurnProb","Risk",
                 "AvgTrips","AvgSpentPerWeek","DaysSinceLastTrip"]
    hist_df = pd.DataFrame(st.session_state.history, columns=hist_cols)
    st.dataframe(hist_df, use_container_width=True)

    csv_buf = BytesIO()
    hist_df.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download History CSV", data=csv_buf.getvalue(),
                       file_name="churn_prediction_history.csv", mime="text/csv")

    if FPDF_AVAILABLE:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 8, "Churn Prediction History", ln=True, align="C")
        pdf.ln(4)
        for _, row in hist_df.iterrows():
            line = " | ".join(map(str, row.values))
            pdf.multi_cell(0, 6, line)
        pdf_bytes = pdf.output(dest="S").encode("latin-1", "ignore")
        st.download_button("⬇️ Download History PDF", data=pdf_bytes,
                           file_name="churn_prediction_history.pdf", mime="application/pdf")
else:
    st.info("No predictions yet. Use the sidebar to run a churn prediction.")

st.caption(f"Data source: **{used_csv}** (use Static tab for dataset charts, Dynamic tab for live predictions).")
