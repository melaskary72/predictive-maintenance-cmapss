"""Sliding-window construction for sequence models and tabular feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    PREDICTION_HORIZONS,
    WINDOW_SIZE,
    WINDOW_STRIDE,
)


def build_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> dict[str, np.ndarray]:
    """Construct fixed-length windows per engine.

    For engines shorter than window_size, left-pad with the first available
    cycle's values repeated.

    Parameters
    ----------
    df : DataFrame sorted by (unit, cycle), already preprocessed and labeled.
    feature_cols : columns to include as features.
    window_size : window length in cycles.
    stride : stride between window start indices.

    Returns
    -------
    Dict with keys:
      X : (n_windows, window_size, n_features)
      y_rul : (n_windows,) clipped RUL at the window's last cycle
      y_class_h{H} : (n_windows,) binary failure labels for each H
      unit_ids : (n_windows,) the engine each window came from
      last_cycles : (n_windows,) the last cycle index of each window
    """
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    n_features = len(feature_cols)

    X_list: list[np.ndarray] = []
    y_rul_list: list[float] = []
    y_class_lists: dict[int, list[int]] = {h: [] for h in PREDICTION_HORIZONS}
    unit_list: list[int] = []
    last_cycle_list: list[int] = []

    for unit, group in df.groupby("unit", sort=True):
        features = group[feature_cols].values  # (n_cycles, n_features)
        ruls = group["rul_clipped"].values
        cycles = group["cycle"].values
        class_labels = {
            h: group[f"failure_within_{h}"].values for h in PREDICTION_HORIZONS
        }
        n_cycles = features.shape[0]

        if n_cycles < window_size:
            # Left-pad with the first cycle repeated
            pad_count = window_size - n_cycles
            pad = np.repeat(features[0:1], pad_count, axis=0)
            padded = np.vstack([pad, features])
            X_list.append(padded)
            y_rul_list.append(float(ruls[-1]))
            for h in PREDICTION_HORIZONS:
                y_class_lists[h].append(int(class_labels[h][-1]))
            unit_list.append(int(unit))
            last_cycle_list.append(int(cycles[-1]))
        else:
            # Slide windows over the engine's cycles
            for start in range(0, n_cycles - window_size + 1, stride):
                end = start + window_size
                X_list.append(features[start:end])
                y_rul_list.append(float(ruls[end - 1]))
                for h in PREDICTION_HORIZONS:
                    y_class_lists[h].append(int(class_labels[h][end - 1]))
                unit_list.append(int(unit))
                last_cycle_list.append(int(cycles[end - 1]))

    X = np.stack(X_list).astype(np.float32)
    out: dict[str, np.ndarray] = {
        "X": X,
        "y_rul": np.array(y_rul_list, dtype=np.float32),
        "unit_ids": np.array(unit_list, dtype=np.int32),
        "last_cycles": np.array(last_cycle_list, dtype=np.int32),
    }
    for h in PREDICTION_HORIZONS:
        out[f"y_class_h{h}"] = np.array(y_class_lists[h], dtype=np.int8)
    return out


def build_tabular_features(
    windowed: dict[str, np.ndarray], feature_cols: list[str]
) -> pd.DataFrame:
    """Extract hand-engineered features per window for classical models.

    For each sensor in the window, computes: mean, std, min, max, slope, last.

    Returns
    -------
    DataFrame with one row per window, columns named like 'sensor_2_mean',
    plus 'unit_id', 'last_cycle', 'rul_clipped', and 'failure_within_H' columns.
    """
    X = windowed["X"]  # (n_windows, window_size, n_features)
    n_windows, window_size, n_features = X.shape

    # Vectorized statistics over time axis
    means = X.mean(axis=1)
    stds = X.std(axis=1)
    mins = X.min(axis=1)
    maxs = X.max(axis=1)
    lasts = X[:, -1, :]

    # Slope: linear regression coefficient over [0, ..., window_size-1] for each window/feature
    t = np.arange(window_size, dtype=np.float32)
    t_mean = t.mean()
    t_centered = t - t_mean
    t_var = (t_centered ** 2).sum()
    # X centered along time axis
    X_mean = X.mean(axis=1, keepdims=True)
    X_centered = X - X_mean
    # slope = sum_t (t_c * x_c) / sum_t (t_c^2)
    slopes = (t_centered[None, :, None] * X_centered).sum(axis=1) / t_var

    cols: dict[str, np.ndarray] = {}
    for i, name in enumerate(feature_cols):
        cols[f"{name}_mean"] = means[:, i]
        cols[f"{name}_std"] = stds[:, i]
        cols[f"{name}_min"] = mins[:, i]
        cols[f"{name}_max"] = maxs[:, i]
        cols[f"{name}_slope"] = slopes[:, i]
        cols[f"{name}_last"] = lasts[:, i]

    df = pd.DataFrame(cols)
    df["unit_id"] = windowed["unit_ids"]
    df["last_cycle"] = windowed["last_cycles"]
    df["rul_clipped"] = windowed["y_rul"]
    for key in windowed:
        if key.startswith("y_class_h"):
            h = key.replace("y_class_h", "")
            df[f"failure_within_{h}"] = windowed[key]
    return df
