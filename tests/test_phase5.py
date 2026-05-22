"""Smoke tests for Phase 5 (horizon sweep + AI4I)."""
from __future__ import annotations

import pickle

import pandas as pd
import pytest

from src.config import PROJECT_ROOT

PHASE5_DIR = PROJECT_ROOT / "results" / "models" / "phase5"
PHASE5_AI4I_DIR = PROJECT_ROOT / "results" / "models" / "phase5_ai4i"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def test_horizon_sweep_table():
    p = TABLES_DIR / "horizon_sweep.csv"
    if not p.exists():
        pytest.skip("Run scripts/horizon_sweep.py first")
    df = pd.read_csv(p)
    assert len(df) >= 9  # 3 horizons x at least 3 model families
    assert set(df["horizon"].unique()) == {10, 30, 50}


def test_horizon_sweep_xgb_models():
    if not (PHASE5_DIR / "xgb_h10.pkl").exists():
        pytest.skip()
    for h in [10, 30, 50]:
        p = PHASE5_DIR / f"xgb_h{h}.pkl"
        assert p.exists(), f"Missing {p}"


def test_ai4i_results_table():
    p = TABLES_DIR / "ai4i_results.csv"
    if not p.exists():
        pytest.skip("Run scripts/train_ai4i.py first")
    df = pd.read_csv(p)
    assert len(df) >= 1
    assert "ai4i_2020" in df["dataset"].values


def test_ai4i_model_loads():
    p = PHASE5_AI4I_DIR / "xgb_ai4i.pkl"
    if not p.exists():
        pytest.skip()
    with open(p, "rb") as f:
        model = pickle.load(f)
    assert hasattr(model, "predict_proba")
