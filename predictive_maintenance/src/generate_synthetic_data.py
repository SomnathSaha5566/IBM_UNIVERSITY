"""Generate a synthetic dataset with the same schema as the Kaggle
'Machine Predictive Maintenance Classification' dataset (AI4I 2020 style).

Use ONLY to smoke-test the pipeline when the real CSV isn't available.
Failure rules follow the AI4I 2020 description.
"""
import argparse
import numpy as np
import pandas as pd


def generate(n=10000, seed=42):
    rng = np.random.default_rng(seed)
    typ = rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1])
    air = rng.normal(300, 2, n)
    proc = air + 10 + rng.normal(0, 1, n)
    rpm = rng.normal(1540, 180, n).clip(1100, 2900)
    torque = rng.normal(40, 10, n).clip(3, 77)
    wear = rng.integers(0, 254, n).astype(float)

    omega = rpm * 2 * np.pi / 60
    power = torque * omega
    strain = wear * torque
    thr = np.select([typ == "L", typ == "M"], [11000, 12000], default=13000)

    hdf = ((proc - air) < 8.6) & (rpm < 1380)
    pwf = (power < 3500) | (power > 9000)
    osf = strain > thr
    twf = (wear >= 200) & (rng.random(n) < 0.12)
    rnf = rng.random(n) < 0.001

    label = np.full(n, "No Failure", dtype=object)
    label[rnf] = "Random Failures"
    label[twf] = "Tool Wear Failure"
    label[hdf] = "Heat Dissipation Failure"
    label[pwf] = "Power Failure"
    label[osf] = "Overstrain Failure"

    return pd.DataFrame({
        "UDI": np.arange(1, n + 1),
        "Product ID": [f"{t}{47000 + i}" for i, t in enumerate(typ)],
        "Type": typ,
        "Air temperature [K]": air.round(1),
        "Process temperature [K]": proc.round(1),
        "Rotational speed [rpm]": rpm.round(0).astype(int),
        "Torque [Nm]": torque.round(1),
        "Tool wear [min]": wear.astype(int),
        "Target": (label != "No Failure").astype(int),
        "Failure Type": label,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/predictive_maintenance.csv")
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    df = generate(a.n)
    df.to_csv(a.out, index=False)
    print(df["Failure Type"].value_counts())
    print("Saved", a.out)
