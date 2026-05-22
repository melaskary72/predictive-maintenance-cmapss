"""AI4I 2020 Predictive Maintenance Dataset loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import RAW_DIR

AI4I_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
AI4I_DIR = RAW_DIR / "ai4i"


def download_ai4i(force: bool = False) -> Path:
    """Download AI4I 2020 dataset CSV. Idempotent."""
    AI4I_DIR.mkdir(parents=True, exist_ok=True)
    target = AI4I_DIR / "ai4i2020.csv"
    if target.exists() and not force:
        return target
    print(f"Downloading AI4I 2020 dataset from {AI4I_URL}")
    resp = requests.get(AI4I_URL, timeout=60)
    resp.raise_for_status()
    target.write_bytes(resp.content)
    print(f"Saved to {target} ({target.stat().st_size} bytes)")
    return target


def load_ai4i() -> pd.DataFrame:
    """Load the AI4I 2020 CSV.

    Columns:
      UDI, Product ID, Type, Air temperature [K], Process temperature [K],
      Rotational speed [rpm], Torque [Nm], Tool wear [min],
      Machine failure (binary target),
      TWF, HDF, PWF, OSF, RNF (specific failure modes)

    Returns the DataFrame with columns renamed to safe Python identifiers.
    """
    path = download_ai4i()
    df = pd.read_csv(path)
    rename = {
        "Air temperature [K]": "air_temp_k",
        "Process temperature [K]": "process_temp_k",
        "Rotational speed [rpm]": "rot_speed_rpm",
        "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "failure",
        "Product ID": "product_id",
    }
    df = df.rename(columns=rename)
    return df


def prepare_ai4i_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build X, y arrays and feature names for AI4I modeling.

    Drops UDI, Product ID, and the per-failure-mode flags (TWF, HDF, PWF, OSF, RNF)
    since those are essentially output columns. One-hot encodes 'Type'.
    """
    df = df.copy()
    drop_cols = ["UDI", "product_id", "TWF", "HDF", "PWF", "OSF", "RNF"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = pd.get_dummies(df, columns=["Type"], drop_first=False, dtype=np.float64)
    y = df["failure"].values.astype(np.int8)
    X_df = df.drop(columns=["failure"])
    feature_names = list(X_df.columns)
    X = X_df.values.astype(np.float64)
    return X, y, feature_names
