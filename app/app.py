"""
CICIDS2017 — Network Intrusion Detection Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_best_model.joblib"
ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.joblib"

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


@st.cache_resource
def get_explainer():
    model, _ = load_model()
    return shap.TreeExplainer(model)


# ── Page config ──
st.set_page_config(page_title="NIDS Dashboard", page_icon="🛡️", layout="wide")

# ── Sidebar ──
with st.sidebar:
    st.title("🛡️ NIDS Dashboard")
    st.caption("CICIDS2017 · XGBoost")
    st.divider()
    uploaded_file = st.file_uploader("Upload network traffic CSV", type=["csv"])
    st.divider()
    st.markdown(
        "**How to use:**\n"
        "1. Upload a CSV\n"
        "2. View predictions below\n"
        "3. Select a threat to see AI explanation"
    )

# ── Main ──
st.title("Network Intrusion Detection System")
st.write("Upload network traffic data to classify flows using the trained XGBoost model.")
st.divider()

# Load data
data = None
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

if data is not None:
    model, le = load_model()

    # Validate columns
    missing = [c for c in EXPECTED_FEATURES if c not in data.columns]
    if missing:
        st.error(f"Missing {len(missing)} required column(s): {', '.join(missing[:5])}")
        st.stop()

    # Preprocess datatypes to prevent XGBoost/SHAP errors from dirty CSVs
    X = data[EXPECTED_FEATURES].copy()
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
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
        display_df = filtered[show_cols].reset_index()
        display_df.rename(columns={"index": "Row Index"}, inplace=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── SHAP Explanation ──
    st.divider()
    st.subheader("AI Explanation (SHAP)")
    st.write("Select a flow to see why the model made its prediction.")

    # Let user pick a row to explain
    row_idx = st.number_input(
        "Row index to explain",
        min_value=0,
        max_value=len(results) - 1,
        value=0,
        step=1,
    )

    if st.button("Explain Prediction"):
        row = X.iloc[[row_idx]]
        pred_label = results.iloc[row_idx]["Prediction"]
        pred_conf = results.iloc[row_idx]["Confidence (%)"]
        pred_class_idx = int(y_pred[row_idx])

        st.info(f"**Row {row_idx}** → Predicted: **{pred_label}** (Confidence: {pred_conf}%)")

        with st.spinner("Calculating SHAP values..."):
            explainer = get_explainer()
            shap_values = explainer.shap_values(row)

        # shap_values shape depends on SHAP version:
        #   - Old: list of arrays, one per class (each shape (1, 52))
        #   - New: single ndarray of shape (1, 52, n_classes)
        if isinstance(shap_values, list):
            sv = shap_values[pred_class_idx][0]
            base = explainer.expected_value[pred_class_idx]
        elif shap_values.ndim == 3:
            sv = shap_values[0, :, pred_class_idx]
            base = explainer.expected_value[pred_class_idx]
        else:
            sv = shap_values[0]
            base = explainer.expected_value

        # Build SHAP Explanation object for waterfall plot
        explanation = shap.Explanation(
            values=sv,
            base_values=float(base),
            data=row.values[0],
            feature_names=EXPECTED_FEATURES,
        )

        # Show top 15 features
        top_k = 15
        top_indices = np.argsort(np.abs(sv))[::-1][:top_k]

        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=top_k, show=False)
        plt.title(f"SHAP Explanation — Predicted: {pred_label}", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Also show top features as a simple table
        st.write("**Top contributing features:**")
        feat_df = pd.DataFrame({
            "Feature": [EXPECTED_FEATURES[i] for i in top_indices],
            "Value": [float(row.values[0][i]) for i in top_indices],
            "SHAP Impact": [round(float(sv[i]), 4) for i in top_indices],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ── Full results ──
    with st.expander("View All Predictions"):
        st.dataframe(results, use_container_width=True)

    # Download
    csv_out = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions CSV", csv_out,
                       file_name="nids_predictions.csv", mime="text/csv")

else:
    st.info("👈 Upload a CSV file to get started.")
