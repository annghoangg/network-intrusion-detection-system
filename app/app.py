"""
CICIDS2017 — Network Intrusion Detection Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_best_model.joblib"
ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.joblib"
SAMPLE_PATH = Path(__file__).resolve().parent / "sample_test_data.csv"

EXPECTED_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Length of Fwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min", "Bwd PSH Flags", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "PSH Flag Count",
    "ACK Flag Count", "Average Packet Size", "Subflow Fwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
    "Active Max", "Active Min", "Idle Mean", "Idle Max", "Idle Min",
]


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    return model, le


# ── Page config ──
st.set_page_config(page_title="NIDS Dashboard", page_icon="🛡️", layout="wide")

# ── Sidebar ──
with st.sidebar:
    st.title("🛡️ NIDS Dashboard")
    st.caption("CICIDS2017 · XGBoost")
    st.divider()
    uploaded_file = st.file_uploader("Upload network traffic CSV", type=["csv"])
    use_sample = st.button("Use Sample Data")
    st.divider()
    st.markdown(
        "**How to use:**\n"
        "1. Upload a CSV or click *Use Sample Data*\n"
        "2. View predictions below"
    )

# ── Main ──
st.title("Network Intrusion Detection System")
st.write("Upload network traffic data to classify flows using the trained XGBoost model.")
st.divider()

# Load data
data = None
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
elif use_sample:
    data = pd.read_csv(SAMPLE_PATH)

if data is not None:
    model, le = load_model()

    # Validate columns
    missing = [c for c in EXPECTED_FEATURES if c not in data.columns]
    if missing:
        st.error(f"Missing {len(missing)} required column(s): {', '.join(missing[:5])}")
        st.stop()

    X = data[EXPECTED_FEATURES]

    # Predict
    with st.spinner("Running predictions..."):
        y_pred = model.predict(X)
        y_labels = le.inverse_transform(y_pred)
        y_proba = model.predict_proba(X)
        confidence = np.max(y_proba, axis=1)

    results = data.copy()
    results["Prediction"] = y_labels
    results["Confidence (%)"] = np.round(confidence * 100, 2)

    total = len(results)
    benign = int((results["Prediction"] == "BENIGN").sum())
    attacks = total - benign

    # ── Overview metrics ──
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Flows", f"{total:,}")
    col2.metric("Benign", f"{benign:,}")
    col3.metric("Attacks", f"{attacks:,}")
    col4.metric("Attack Rate", f"{attacks / total * 100:.1f}%")

    # ── Prediction distribution ──
    st.subheader("Prediction Distribution")
    dist = results["Prediction"].value_counts()
    st.bar_chart(dist)

    # ── Detected threats ──
    st.subheader("Detected Threats")
    attack_df = results[results["Prediction"] != "BENIGN"]

    if attack_df.empty:
        st.success("No threats detected — all flows are BENIGN.")
    else:
        attack_types = sorted(attack_df["Prediction"].unique())
        selected = st.multiselect("Filter by attack type:", attack_types, default=attack_types)
        filtered = attack_df[attack_df["Prediction"].isin(selected)]

        show_cols = ["Prediction", "Confidence (%)", "Destination Port",
                     "Flow Duration", "Flow Bytes/s", "Total Fwd Packets"]
        show_cols = [c for c in show_cols if c in filtered.columns]
        st.dataframe(filtered[show_cols].reset_index(drop=True), use_container_width=True)

    # ── Full results ──
    with st.expander("View All Predictions"):
        st.dataframe(results, use_container_width=True)

    # Download
    csv_out = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions CSV", csv_out,
                       file_name="nids_predictions.csv", mime="text/csv")

else:
    st.info("👈 Upload a CSV file or click **Use Sample Data** to get started.")
