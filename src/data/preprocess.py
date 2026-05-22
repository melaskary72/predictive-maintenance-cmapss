"""Preprocessing for C-MAPSS: RUL labels, binary labels, normalization, sensor screening."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import (
    ALL_SENSOR_COLS,
    N_OP_REGIMES,
    OP_SETTING_COLS,
    PREDICTION_HORIZONS,
    RANDOM_SEED,
    RUL_CEILING,
    SENSOR_VARIANCE_THRESHOLD,
)


def add_rul_to_train(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RUL for training data: max_cycle_for_unit minus current cycle.

    Returns the input DataFrame with two new columns:
      - rul: raw remaining useful life
      - rul_clipped: rul clipped at RUL_CEILING
    """
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = max_cycle - df["cycle"]
    df["rul_clipped"] = df["rul"].clip(upper=RUL_CEILING)
    return df


def add_rul_to_test(df: pd.DataFrame, true_rul_at_last: np.ndarray) -> pd.DataFrame:
    """Compute RUL for test data using the truncation-point ground truth.

    For test engine u, true RUL at its last observed cycle is given. Earlier
    cycles have RUL = true_RUL_at_last + (last_cycle - current_cycle).

    Parameters
    ----------
    df : test data DataFrame
    true_rul_at_last : 1D array, true_rul_at_last[i] is RUL of unit (i+1)
                       at its last cycle.
    """
    df = df.copy()
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    # unit numbers in test files are 1-indexed
    rul_at_last = pd.Series(
        true_rul_at_last[df["unit"].values - 1], index=df.index
    )
    df["rul"] = rul_at_last + (last_cycle - df["cycle"])
    df["rul_clipped"] = df["rul"].clip(upper=RUL_CEILING)
    return df


def add_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary failure labels at each prediction horizon.

    For each horizon H, adds a column 'failure_within_H' that is 1 if
    rul <= H else 0.
    """
    df = df.copy()
    for h in PREDICTION_HORIZONS:
        df[f"failure_within_{h}"] = (df["rul"] <= h).astype(np.int8)
    return df


def identify_flat_sensors(
    df_train: pd.DataFrame, threshold: float = SENSOR_VARIANCE_THRESHOLD
) -> list[str]:
    """Return sensor column names with training-set variance below threshold."""
    variances = df_train[ALL_SENSOR_COLS].var()
    return variances[variances < threshold].index.tolist()


def fit_simple_scaler(
    df_train: pd.DataFrame, sensor_cols: list[str]
) -> StandardScaler:
    """Fit a single StandardScaler on training sensors. For FD001, FD003."""
    scaler = StandardScaler()
    scaler.fit(df_train[sensor_cols].values)
    return scaler


def apply_simple_scaler(
    df: pd.DataFrame, sensor_cols: list[str], scaler: StandardScaler
) -> pd.DataFrame:
    """Apply a fitted simple scaler to a DataFrame in place (returns copy)."""
    df = df.copy()
    df[sensor_cols] = scaler.transform(df[sensor_cols].values).astype(np.float64)
    return df


def fit_regime_kmeans(
    df_train: pd.DataFrame, n_regimes: int = N_OP_REGIMES
) -> KMeans:
    """Cluster operational settings into discrete regimes. For FD002, FD004."""
    kmeans = KMeans(
        n_clusters=n_regimes, random_state=RANDOM_SEED, n_init=10
    )
    kmeans.fit(df_train[OP_SETTING_COLS].values)
    return kmeans


def assign_regimes(df: pd.DataFrame, kmeans: KMeans) -> pd.DataFrame:
    """Add a 'regime' column to df based on a fitted KMeans model."""
    df = df.copy()
    df["regime"] = kmeans.predict(df[OP_SETTING_COLS].values)
    return df


def fit_regime_scalers(
    df_train: pd.DataFrame, sensor_cols: list[str]
) -> dict[int, StandardScaler]:
    """Fit one StandardScaler per regime on training data.

    Assumes df_train already has a 'regime' column from assign_regimes.
    """
    scalers: dict[int, StandardScaler] = {}
    for regime in sorted(df_train["regime"].unique()):
        mask = df_train["regime"] == regime
        scaler = StandardScaler()
        scaler.fit(df_train.loc[mask, sensor_cols].values)
        scalers[int(regime)] = scaler
    return scalers


def apply_regime_scalers(
    df: pd.DataFrame,
    sensor_cols: list[str],
    scalers: dict[int, StandardScaler],
) -> pd.DataFrame:
    """Apply per-regime scalers to df (which must have a 'regime' column)."""
    df = df.copy()
    # Ensure target columns are float so scaled values can be assigned without
    # triggering a lossy dtype error on integer-valued raw sensors.
    df[sensor_cols] = df[sensor_cols].astype(np.float64)
    for regime, scaler in scalers.items():
        mask = df["regime"] == regime
        if mask.any():
            df.loc[mask, sensor_cols] = scaler.transform(
                df.loc[mask, sensor_cols].values
            )
    return df
