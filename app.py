"""Streamlit dashboard for the simulated voltage-classification project."""
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(page_title="AI Voltage Monitor", page_icon="⚡", layout="centered")
st.title("⚡ AI-Based Voltage Condition Monitor")
st.caption("Educational simulation of a 12 V DC system — not a safety protection device.")

model_path = Path("voltage_model.pkl")
if not model_path.exists():
    st.error("Model not found. Run `python train_model.py` in this folder first.")
    st.stop()
model = joblib.load(model_path)

st.info("Illustrative thresholds: below 10.8 V = undervoltage; 10.8–13.2 V = normal; above 13.2 V = overvoltage.")
voltage = st.number_input("Voltage (V)", min_value=0.0, max_value=30.0, value=12.0, step=0.1)
current = st.number_input("Current (A)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
load = st.slider("Load (%)", min_value=0, max_value=100, value=50)

if st.button("Classify voltage", type="primary"):
    sample = pd.DataFrame([{"voltage": voltage, "current": current, "load": load}])
    predicted = model.predict(sample)[0]
    st.subheader("Machine-learning prediction")
    st.write(f"**{predicted}**")
    # Independent threshold rule makes the demo warning transparent.
    if voltage < 10.8:
        st.error("WARNING: Undervoltage according to the defined project threshold.")
    elif voltage > 13.2:
        st.error("WARNING: Overvoltage according to the defined project threshold.")
    else:
        st.success("Voltage is within the defined normal range.")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample)[0]
        st.write("Model class probabilities")
        st.bar_chart(pd.DataFrame({"Probability": probs}, index=model.classes_))
    st.caption("This simulated model is not validated for real electrical protection or deployment.")
