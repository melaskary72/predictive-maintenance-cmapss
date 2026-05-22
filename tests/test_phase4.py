"""Smoke tests for Phase 4 Transformer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PROJECT_ROOT

PHASE4_DIR = PROJECT_ROOT / "results" / "models" / "phase4"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


@pytest.fixture(scope="module")
def has_artifacts() -> bool:
    return (PHASE4_DIR / "transformer_final.pt").exists()


def test_phase4_results_table(has_artifacts):
    if not has_artifacts:
        pytest.skip("Run scripts/train_transformer.py first")
    p = TABLES_DIR / "phase4_results.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) >= 4  # transformer + transformer_with_aux for both tasks


def test_master_comparison_exists(has_artifacts):
    p = TABLES_DIR / "master_comparison.csv"
    if not p.exists():
        pytest.skip("Run scripts/run_full_ensemble.py first")
    df = pd.read_csv(p)
    assert "transformer" in df["model"].values


def test_transformer_predictions_shape(has_artifacts):
    if not has_artifacts:
        pytest.skip()
    arr = np.load(PHASE4_DIR / "transformer_rul_test.npy")
    assert arr.ndim == 1
    assert arr.shape[0] > 0


def test_transformer_clf_in_range(has_artifacts):
    if not has_artifacts:
        pytest.skip()
    arr = np.load(PHASE4_DIR / "transformer_clf_test.npy")
    assert arr.min() >= 0
    assert arr.max() <= 1


def test_transformer_forward_shapes():
    """Verify Transformer forward passes produce correct output shapes."""
    import sys

    if "xgboost" in sys.modules:
        pytest.skip(
            "xgboost is loaded in this process; deep models would deadlock "
            "due to OpenMP runtime conflict."
        )

    import torch
    from src.models.deep import TransformerRegressor

    X = torch.randn(4, 30, 14)
    aux = torch.randn(4, 84)

    tx = TransformerRegressor(
        n_features=14, d_model=32, nhead=4, num_layers=2,
        dim_feedforward=64, dropout=0.1,
    )
    r, c = tx(X)
    assert r.shape == (4,)
    assert c.shape == (4,)

    tx_aux = TransformerRegressor(
        n_features=14, d_model=32, nhead=4, num_layers=2,
        dim_feedforward=64, dropout=0.1, aux_features=84,
    )
    r, c = tx_aux(X, aux)
    assert r.shape == (4,)
    assert c.shape == (4,)
