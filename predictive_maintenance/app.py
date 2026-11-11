"""Streamlit dashboard:  streamlit run app.py"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import predict  # noqa: E402

st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️")
st.title("🛠️ Predictive Maintenance Assistant")
st.caption("Enter live machine readings to predict the failure type.")

if not os.path.exists("models/model.joblib"):
    st.error("No trained model found. Run `python src/train.py` first.")
    st.stop()

c1, c2 = st.columns(2)
typ = c1.selectbox("Product quality type", ["L", "M", "H"])
air = c1.number_input("Air temperature [K]", 290.0, 310.0, 300.0, 0.1)
proc = c1.number_input("Process temperature [K]", 300.0, 320.0, 310.0, 0.1)
rpm = c2.number_input("Rotational speed [rpm]", 1000, 3000, 1500, 10)
torque = c2.number_input("Torque [Nm]", 0.0, 100.0, 40.0, 0.5)
wear = c2.number_input("Tool wear [min]", 0, 300, 100, 1)

if st.button("Predict", type="primary"):
    label, proba = predict({
        "Type": typ, "Air temperature [K]": air, "Process temperature [K]": proc,
        "Rotational speed [rpm]": rpm, "Torque [Nm]": torque, "Tool wear [min]": wear,
    })
    (st.success if label == "No Failure" else st.error)(f"Prediction: **{label}**")
    st.bar_chart(pd.Series(proba, name="probability"))
    advice = {
        "Heat Dissipation Failure": "Check coolant flow and cooling fins; reduce load until the process/air temperature gap recovers.",
        "Power Failure": "Inspect motor, supply and drive; torque x speed is outside the safe power band.",
        "Overstrain Failure": "Reduce torque; tool wear x torque is above the limit for this product type.",
        "Tool Wear Failure": "Replace or re-sharpen the tool soon.",
        "Random Failures": "No clear cause - schedule a general inspection.",
    }
    if label in advice:
        st.warning("Suggested action: " + advice[label])
