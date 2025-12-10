# wildlife_dashboard.py
# Streamlit dashboard for Wildlife Poaching Prediction System
# Features:
# - Geospatial map showing predicted risk zones (red/yellow/green markers)
# - Live data feed panel (simulated streaming updates)
# - Model summary and simple controls
#
# Run with:
#     streamlit run wildlife_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import pydeck as pdk


# --- Page Config ---
st.set_page_config(page_title="Wildlife Poaching Dashboard", layout="wide")

# --- File Paths ---
DATA_DIR = "data"
PRED_CSV = os.path.join(DATA_DIR, "predicted_poaching.csv")
RISK_DIR = os.path.join(DATA_DIR, "risk_predictions")
POACH_CSV = "poaching_data.csv"


# --- Utility Functions ---
def risk_to_color(score: float):
    """Convert risk score to RGB color"""
    if pd.isna(score):
        return [128, 128, 128]
    if score >= 0.66:
        return [220, 20, 60]   # red
    elif score >= 0.33:
        return [255, 215, 0]   # yellow
    else:
        return [34, 139, 34]   # green


@st.cache_data(ttl=5)
def load_predictions():
    """Load model predictions or fallback to synthetic data"""
    # Try main predicted CSV
    if os.path.exists(PRED_CSV):
        try:
            df = pd.read_csv(PRED_CSV)
            st.info(f"Loaded predictions from {PRED_CSV}")
            return df
        except Exception as e:
            st.warning(f"Failed to read {PRED_CSV}: {e}")

    # Try parquet output folder
    if os.path.isdir(RISK_DIR):
        parquets = glob.glob(os.path.join(RISK_DIR, "*.parquet"))
        if parquets:
            latest = max(parquets, key=os.path.getmtime)
            try:
                df = pd.read_parquet(latest)
                st.info(f"Loaded predictions from {latest}")
                return df
            except Exception as e:
                st.warning(f"Failed to read parquet {latest}: {e}")

    # Fallback to poaching_data.csv
    if os.path.exists(POACH_CSV):
        df = pd.read_csv(POACH_CSV)
        temp_norm = (df["temperature"] - df["temperature"].min()) / (
            df["temperature"].max() - df["temperature"].min() + 1e-9
        )
        dens_norm = (df["animal_density"] - df["animal_density"].min()) / (
            df["animal_density"].max() - df["animal_density"].min() + 1e-9
        )
        df["risk_score"] = 0.6 * temp_norm + 0.4 * dens_norm
        st.info(f"Loaded base data from {POACH_CSV} and synthesized risk_score")
        return df

    # Generate sample data if no files found
    n = 20
    lats = np.random.uniform(-1.6, -1.4, size=n)
    lons = np.random.uniform(36.65, 36.72, size=n)
    df = pd.DataFrame({
        "id": [f"S{i+1:03d}" for i in range(n)],
        "lat": lats,
        "lon": lons,
        "temperature": np.random.uniform(30, 40, size=n),
        "animal_density": np.random.uniform(5, 15, size=n)
    })
    temp_norm = (df["temperature"] - df["temperature"].min()) / (
        df["temperature"].max() - df["temperature"].min() + 1e-9
    )
    dens_norm = (df["animal_density"] - df["animal_density"].min()) / (
        df["animal_density"].max() - df["animal_density"].min() + 1e-9
    )
    df["risk_score"] = 0.6 * temp_norm + 0.4 * dens_norm
    st.info("No data files found — using generated sample data")
    return df


def simulate_live_update(df: pd.DataFrame, jitter=0.0005):
    """Simulate new live data by slightly changing lat/lon and risk scores"""
    df2 = df.copy()
    if "lat" in df2.columns and "lon" in df2.columns:
        df2["lat"] += np.random.normal(scale=jitter, size=len(df2))
        df2["lon"] += np.random.normal(scale=jitter, size=len(df2))
    if "risk_score" in df2.columns:
        df2["risk_score"] = (
            df2["risk_score"] + np.random.normal(scale=0.02, size=len(df2))
        ).clip(0, 1)
    return df2


# --- Dashboard Layout ---
st.title("🦏 Wildlife Poaching Prediction — Dashboard")
st.markdown("Analyze predicted poaching risk zones and observe live updates in real-time.")

col1, col2 = st.columns((2, 1))

with col2:
    st.header("Controls & Status")
    refresh_interval = st.slider("Auto-refresh interval (seconds)", 2, 30, 5)
    simulate = st.checkbox("Simulate live feed", value=True)
    n_top = st.number_input("Show top N high-risk points", 1, 500, 10)
    st.write("---")
    st.write("Data sources checked:")
    st.write(f"• {PRED_CSV} (CSV)")
    st.write(f"• {RISK_DIR} (Parquet dir)")
    st.write(f"• {POACH_CSV} (Fallback CSV)")

# Load initial data
base_df = load_predictions()

# Normalize column names
if "lat" not in base_df.columns and "latitude" in base_df.columns:
    base_df = base_df.rename(columns={"latitude": "lat", "longitude": "lon"})

# Ensure 'id' column
if "id" not in base_df.columns:
    base_df["id"] = [f"pt_{i+1}" for i in range(len(base_df))]

map_slot = col1.container()

# --- Model Summary ---
with col2:
    st.header("Model Summary")
    if "risk_score" in base_df.columns:
        avg_risk = base_df["risk_score"].mean()
        max_risk = base_df["risk_score"].max()
        st.metric("Average risk score", f"{avg_risk:.2f}")
        st.metric("Max risk score", f"{max_risk:.2f}")
    else:
        st.write("No risk_score column available in loaded data")

# --- Auto Refresh ---
count = st.query_params.get("refresh_count", [0])  # ✅ Updated line

if "live_df" not in st.session_state:
    st.session_state.live_df = base_df.copy()

if simulate:
    st_autorefresh(interval=refresh_interval * 1000, key="autorefresh")
    st.session_state.live_df = simulate_live_update(st.session_state.live_df)

live_df = st.session_state.live_df

# --- Map Visualization ---
if not ("lat" in live_df.columns and "lon" in live_df.columns):
    st.error("Loaded data does not contain latitude/longitude columns.")
else:
    plot_df = live_df.dropna(subset=["lat", "lon"]).copy()
    if "risk_score" not in plot_df.columns:
        plot_df["risk_score"] = 0.0
    plot_df["marker_size"] = (plot_df["risk_score"].fillna(0) * 30) + 5
    plot_df["color"] = plot_df["risk_score"].fillna(0).apply(lambda x: risk_to_color(x))
    top_df = plot_df.sort_values("risk_score", ascending=False).head(int(n_top))

    st.subheader("Map — Predicted Risk Zones")
    midpoint = (plot_df["lat"].mean(), plot_df["lon"].mean())

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius="marker_size",
        pickable=True,
    )

    view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=12)
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "ID: {id}\\nRisk: {risk_score:.2f}"},
    )

    map_slot.pydeck_chart(deck)

    st.markdown("---")
    st.subheader("Top High-Risk Locations")
    st.dataframe(
        top_df[
            [c for c in ["id", "lat", "lon", "risk_score", "temperature", "animal_density"] if c in top_df.columns]
        ]
    )

# --- Live Feed ---
with col2:
    st.header("Live Data Feed")
    if st.button("Force update / refresh"):
        st.session_state.live_df = simulate_live_update(st.session_state.live_df)
        st.rerun()

    st.write("Latest feed (sample):")
    st.dataframe(live_df.sample(min(10, len(live_df))).reset_index(drop=True))

st.markdown("---")
st.caption(
    "This dashboard reads predictions from data/predicted_poaching.csv or data/risk_predictions/*.parquet. "
    "If none found, it falls back to poaching_data.csv or generates sample data."
)
