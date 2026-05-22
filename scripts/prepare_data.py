"""Top-level data preparation pipeline.

Reads raw C-MAPSS files, applies preprocessing per subset, builds windows
and tabular features, splits engines, and saves everything to data/processed/.

Run from project root:
    uv run python scripts/prepare_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    ALL_SENSOR_COLS,
    CMAPSS_SUBSETS,
    INTERIM_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
)
from src.data.load import load_cmapss_rul, load_cmapss_test, load_cmapss_train
from src.data.preprocess import (
    add_binary_labels,
    add_rul_to_test,
    add_rul_to_train,
    apply_regime_scalers,
    apply_simple_scaler,
    assign_regimes,
    fit_regime_kmeans,
    fit_regime_scalers,
    fit_simple_scaler,
    identify_flat_sensors,
)
from src.data.splits import make_engine_splits
from src.data.windows import build_tabular_features, build_windows
from src.utils.seeds import set_seeds


# Subsets with multiple operating regimes
MULTI_REGIME_SUBSETS = {"FD002", "FD004"}


def prepare_subset(subset: str) -> dict:
    """Run the full pipeline for one C-MAPSS subset."""
    print(f"\n=== Preparing subset {subset} ===")

    # Load
    train_raw = load_cmapss_train(subset)
    test_raw = load_cmapss_test(subset)
    test_rul = load_cmapss_rul(subset)
    print(f"  Loaded train: {train_raw.shape}, test: {test_raw.shape}")

    # Add labels
    train = add_rul_to_train(train_raw)
    test = add_rul_to_test(test_raw, test_rul)
    train = add_binary_labels(train)
    test = add_binary_labels(test)

    # Identify and drop flat sensors based on training data
    flat_sensors = identify_flat_sensors(train)
    sensor_cols = [s for s in ALL_SENSOR_COLS if s not in flat_sensors]
    print(f"  Flat sensors dropped: {flat_sensors}")
    print(f"  Sensors kept: {len(sensor_cols)}")

    # Normalize
    if subset in MULTI_REGIME_SUBSETS:
        kmeans = fit_regime_kmeans(train)
        train = assign_regimes(train, kmeans)
        test = assign_regimes(test, kmeans)
        scalers = fit_regime_scalers(train, sensor_cols)
        train = apply_regime_scalers(train, sensor_cols, scalers)
        test = apply_regime_scalers(test, sensor_cols, scalers)
        print(f"  Applied per-regime normalization ({len(scalers)} regimes)")
    else:
        scaler = fit_simple_scaler(train, sensor_cols)
        train = apply_simple_scaler(train, sensor_cols, scaler)
        test = apply_simple_scaler(test, sensor_cols, scaler)
        print("  Applied simple normalization")

    # Save interim parquet
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(INTERIM_DIR / f"{subset}_train.parquet", index=False)
    test.to_parquet(INTERIM_DIR / f"{subset}_test.parquet", index=False)

    # Build windows
    train_windowed = build_windows(train, sensor_cols)
    test_windowed = build_windows(test, sensor_cols)
    print(
        f"  Train windows: {train_windowed['X'].shape}, "
        f"Test windows: {test_windowed['X'].shape}"
    )

    # Engine-level splits
    train_engines, val_engines = make_engine_splits(
        train_windowed["unit_ids"]
    )
    print(
        f"  Train engines: {len(train_engines)}, "
        f"Val engines: {len(val_engines)}, "
        f"Test engines: {len(np.unique(test_windowed['unit_ids']))}"
    )

    # Build tabular features
    train_tabular = build_tabular_features(train_windowed, sensor_cols)
    test_tabular = build_tabular_features(test_windowed, sensor_cols)

    # Save processed arrays
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PROCESSED_DIR / f"{subset}_windows_train.npz",
        train_engine_ids=train_engines,
        val_engine_ids=val_engines,
        sensor_cols=np.array(sensor_cols),
        **train_windowed,
    )
    np.savez_compressed(
        PROCESSED_DIR / f"{subset}_windows_test.npz",
        sensor_cols=np.array(sensor_cols),
        **test_windowed,
    )
    train_tabular.to_parquet(
        PROCESSED_DIR / f"{subset}_tabular_train.parquet", index=False
    )
    test_tabular.to_parquet(
        PROCESSED_DIR / f"{subset}_tabular_test.parquet", index=False
    )

    return {
        "subset": subset,
        "n_train_windows": int(train_windowed["X"].shape[0]),
        "n_test_windows": int(test_windowed["X"].shape[0]),
        "n_features": len(sensor_cols),
        "n_train_engines": int(len(train_engines)),
        "n_val_engines": int(len(val_engines)),
        "n_test_engines": int(np.unique(test_windowed["unit_ids"]).size),
        "flat_sensors_dropped": flat_sensors,
        "sensors_kept": sensor_cols,
    }


def main() -> None:
    set_seeds(RANDOM_SEED)
    summaries = []
    for subset in CMAPSS_SUBSETS:
        summaries.append(prepare_subset(subset))

    # Save a manifest summarizing the pipeline output
    manifest_path = PROCESSED_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n=== Pipeline complete. Manifest at {manifest_path} ===")
    for s in summaries:
        print(
            f"  {s['subset']}: train={s['n_train_windows']} windows, "
            f"test={s['n_test_windows']} windows, "
            f"features={s['n_features']}"
        )


if __name__ == "__main__":
    main()
