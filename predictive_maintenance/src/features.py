"""Shared data loading, cleaning and feature engineering."""
import numpy as np
import pandas as pd

RAW_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Failure Type"
NO_FAILURE = "No Failure"


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the Kaggle CSV and remove known label inconsistencies.

    The dataset contains rows where `Target` (binary) and `Failure Type`
    disagree (e.g. Target=1 but 'No Failure', or 'Random Failures' with
    Target=0). Those labels are unreliable, so they are dropped.
    """
    df = pd.read_csv(path)
    if "Target" in df.columns:
        bad = ((df["Target"] == 1) & (df[TARGET] == NO_FAILURE)) | (
            (df["Target"] == 0) & (df[TARGET] != NO_FAILURE)
        )
        print(f"Dropping {int(bad.sum())} rows with inconsistent labels")
        df = df[~bad]
    return df.reset_index(drop=True)


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Physics-informed features that expose each failure mechanism."""
    X = X.copy()
    X["Type"] = X["Type"].map({"L": 0, "M": 1, "H": 2}).astype(int)
    X["Temp diff [K]"] = X["Process temperature [K]"] - X["Air temperature [K]"]  # heat dissipation
    omega = X["Rotational speed [rpm]"] * 2 * np.pi / 60
    X["Power [W]"] = X["Torque [Nm]"] * omega  # power failure
    X["Strain"] = X["Tool wear [min]"] * X["Torque [Nm]"]  # overstrain failure
    return X


def split_xy(df: pd.DataFrame):
    X = add_features(df[RAW_FEATURES])
    y = df[TARGET]
    return X, y
