"""Sanity tests for the data pipeline. Run with: uv run pytest tests/"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import (
    CMAPSS_SUBSETS,
    PREDICTION_HORIZONS,
    PROCESSED_DIR,
    RUL_CEILING,
    WINDOW_SIZE,
)


@pytest.fixture(scope="module")
def manifest() -> list[dict]:
    path = PROCESSED_DIR / "manifest.json"
    if not path.exists():
        pytest.skip("Run scripts/prepare_data.py first")
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("subset", CMAPSS_SUBSETS)
def test_train_arrays_exist(subset: str) -> None:
    path = PROCESSED_DIR / f"{subset}_windows_train.npz"
    assert path.exists(), f"Missing {path}"


@pytest.mark.parametrize("subset", CMAPSS_SUBSETS)
def test_test_arrays_exist(subset: str) -> None:
    path = PROCESSED_DIR / f"{subset}_windows_test.npz"
    assert path.exists(), f"Missing {path}"


@pytest.mark.parametrize("subset", CMAPSS_SUBSETS)
def test_window_shapes_consistent(subset: str) -> None:
    data = np.load(PROCESSED_DIR / f"{subset}_windows_train.npz")
    n = data["X"].shape[0]
    assert data["X"].shape[1] == WINDOW_SIZE
    assert data["y_rul"].shape == (n,)
    assert data["unit_ids"].shape == (n,)
    for h in PREDICTION_HORIZONS:
        assert data[f"y_class_h{h}"].shape == (n,)


@pytest.mark.parametrize("subset", CMAPSS_SUBSETS)
def test_rul_ceiling_respected(subset: str) -> None:
    data = np.load(PROCESSED_DIR / f"{subset}_windows_train.npz")
    assert data["y_rul"].max() <= RUL_CEILING + 1e-6


@pytest.mark.parametrize("subset", CMAPSS_SUBSETS)
def test_binary_labels_are_zero_one(subset: str) -> None:
    data = np.load(PROCESSED_DIR / f"{subset}_windows_train.npz")
    for h in PREDICTION_HORIZONS:
        labels = data[f"y_class_h{h}"]
        unique = set(np.unique(labels).tolist())
        assert unique.issubset({0, 1}), f"Got values {unique} for h={h}"


@pytest.mark.parametrize("subset", CMAPSS_SUBSETS)
def test_engine_split_no_overlap(subset: str) -> None:
    data = np.load(PROCESSED_DIR / f"{subset}_windows_train.npz")
    train_engines = set(data["train_engine_ids"].tolist())
    val_engines = set(data["val_engine_ids"].tolist())
    assert len(train_engines & val_engines) == 0, "Engines overlap split"


@pytest.mark.parametrize("subset", CMAPSS_SUBSETS)
def test_engine_split_covers_all_units(subset: str) -> None:
    data = np.load(PROCESSED_DIR / f"{subset}_windows_train.npz")
    train_engines = set(data["train_engine_ids"].tolist())
    val_engines = set(data["val_engine_ids"].tolist())
    all_engines = set(np.unique(data["unit_ids"]).tolist())
    assert train_engines | val_engines == all_engines


@pytest.mark.parametrize("subset", CMAPSS_SUBSETS)
def test_normalized_features_roughly_centered(subset: str) -> None:
    """After normalization, training feature means should be near zero
    and stds near one within reasonable tolerance.

    Note: scalers fit on per-cycle data, but windows oversample later cycles
    (each non-boundary cycle appears in WINDOW_SIZE windows under stride=1),
    so windowed-stack means drift slightly from zero. We use generous
    tolerances that catch real normalization bugs without flagging this
    expected effect.

    For multi-regime subsets, per-regime normalization can flatten sensors
    that are non-flat globally (e.g. sensors whose only variation is between
    regimes), producing zero stds — which is correct behavior.
    """
    data = np.load(PROCESSED_DIR / f"{subset}_windows_train.npz")
    X = data["X"]
    means = X.reshape(-1, X.shape[-1]).mean(axis=0)
    stds = X.reshape(-1, X.shape[-1]).std(axis=0)
    is_multi = subset in {"FD002", "FD004"}
    if is_multi:
        # Per-regime scaling can produce zero-variance sensors globally.
        mean_tol = 0.5
        # Stds must be either ~0 (flattened by per-regime mean removal) or
        # within a reasonable band of 1.
        non_flat = stds > 1e-3
        assert np.all(np.abs(means) < mean_tol), f"Means: {means}"
        assert np.all((stds[non_flat] > 0.5) & (stds[non_flat] < 1.5)), (
            f"Stds: {stds}"
        )
    else:
        mean_tol = 0.15
        std_low, std_high = 0.7, 1.3
        assert np.all(np.abs(means) < mean_tol), f"Means: {means}"
        assert np.all((stds > std_low) & (stds < std_high)), f"Stds: {stds}"
