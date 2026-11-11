"""Predict failure type for one machine reading.

Example:
    python src/predict.py --type L --air 298.5 --process 308.6 --rpm 1400 --torque 45 --wear 210
"""
import argparse
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import RAW_FEATURES, add_features  # noqa: E402


def predict(reading: dict, model_path="models/model.joblib"):
    bundle = joblib.load(model_path)
    X = add_features(pd.DataFrame([reading])[RAW_FEATURES])[bundle["features"]]
    model = bundle["model"]
    label = model.predict(X)[0]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0].round(4)))
    return label, proba


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["L", "M", "H"], required=True)
    ap.add_argument("--air", type=float, required=True, help="Air temperature [K]")
    ap.add_argument("--process", type=float, required=True, help="Process temperature [K]")
    ap.add_argument("--rpm", type=float, required=True)
    ap.add_argument("--torque", type=float, required=True)
    ap.add_argument("--wear", type=float, required=True, help="Tool wear [min]")
    a = ap.parse_args()
    label, proba = predict({
        "Type": a.type, "Air temperature [K]": a.air, "Process temperature [K]": a.process,
        "Rotational speed [rpm]": a.rpm, "Torque [Nm]": a.torque, "Tool wear [min]": a.wear,
    })
    print("Predicted:", label)
    for k, v in sorted(proba.items(), key=lambda kv: -kv[1]):
        print(f"  {k:28s} {v:.3f}")
