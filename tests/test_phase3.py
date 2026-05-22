"""Smoke tests for Phase 3 deep models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PROJECT_ROOT

PHASE3_DIR = PROJECT_ROOT / "results" / "models" / "phase3"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


@pytest.fixture(scope="module")
def has_artifacts() -> bool:
    return (PHASE3_DIR / "lstm_final.pt").exists()


def test_phase3_results_table(has_artifacts):
    if not has_artifacts:
        pytest.skip("Run scripts/train_deep.py first")
    p = TABLES_DIR / "phase3_results.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) >= 8


def test_ensemble_results_table(has_artifacts):
    p = TABLES_DIR / "phase3_ensemble_results.csv"
    if not p.exists():
        pytest.skip("Run scripts/run_ensemble.py first")
    df = pd.read_csv(p)
    assert "ensemble_avg_xgb_lstm_cnn" in df["model"].values


def test_lstm_predictions_shape(has_artifacts):
    if not has_artifacts:
        pytest.skip()
    arr = np.load(PHASE3_DIR / "lstm_rul_test.npy")
    assert arr.ndim == 1
    assert arr.shape[0] > 0


def test_cnn_predictions_shape(has_artifacts):
    if not has_artifacts:
        pytest.skip()
    arr = np.load(PHASE3_DIR / "cnn_rul_test.npy")
    assert arr.ndim == 1
    assert arr.shape[0] > 0


def test_clf_probs_in_range(has_artifacts):
    if not has_artifacts:
        pytest.skip()
    for name in ["lstm_clf_test", "cnn_clf_test"]:
        arr = np.load(PHASE3_DIR / f"{name}.npy")
        assert arr.min() >= 0
        assert arr.max() <= 1


def test_deep_model_forward_shapes():
    """Verify model forward passes produce correct output shapes.

    Skipped when xgboost has already been imported in the same process
    (Apple Silicon OpenMP runtime conflict — see test_training_loop_smoke).
    """
    import sys

    if "xgboost" in sys.modules:
        pytest.skip(
            "xgboost is loaded in this process; LSTM forward would deadlock "
            "due to OpenMP runtime conflict. Run this file alone to exercise."
        )

    import torch
    from src.models.deep import DilatedConv1DRegressor, LSTMRegressor

    X = torch.randn(4, 30, 14)
    aux = torch.randn(4, 84)

    lstm = LSTMRegressor(n_features=14, hidden=32, num_layers=2, dropout=0.2)
    r, c = lstm(X)
    assert r.shape == (4,)
    assert c.shape == (4,)

    lstm_aux = LSTMRegressor(
        n_features=14, hidden=32, num_layers=2, dropout=0.2, aux_features=84
    )
    r, c = lstm_aux(X, aux)
    assert r.shape == (4,)
    assert c.shape == (4,)

    cnn = DilatedConv1DRegressor(
        n_features=14, n_channels=32, n_blocks=3, kernel_size=3, dropout=0.2
    )
    r, c = cnn(X)
    assert r.shape == (4,)
    assert c.shape == (4,)

    cnn_aux = DilatedConv1DRegressor(
        n_features=14, n_channels=32, n_blocks=3, kernel_size=3,
        dropout=0.2, aux_features=84,
    )
    r, c = cnn_aux(X, aux)
    assert r.shape == (4,)
    assert c.shape == (4,)


def test_training_loop_smoke():
    """Smoke test the training loop with random data.

    Skipped when xgboost has already been imported in the same process — on
    Apple Silicon, mixing the xgboost and pytorch OpenMP runtimes deadlocks the
    LSTM training. The full training pipeline is validated end-to-end via the
    artifact tests below; this smoke test only adds value when run in isolation.
    """
    import sys

    if "xgboost" in sys.modules:
        pytest.skip(
            "xgboost is loaded in this process; LSTM training would deadlock "
            "due to OpenMP runtime conflict. Run this file alone to exercise."
        )

    import torch
    from torch.utils.data import DataLoader

    from src.models.deep import LSTMRegressor
    from src.models.deep_training import (
        CMAPSSWindowDataset, TrainingConfig, train_deep_model,
    )

    n = 64
    X = np.random.randn(n, 30, 14).astype(np.float32)
    y_rul = np.random.uniform(0, 125, n).astype(np.float32)
    y_clf = np.random.randint(0, 2, n).astype(np.float32)
    ds = CMAPSSWindowDataset(X, y_rul, y_clf)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    device = torch.device("cpu")
    m = LSTMRegressor(n_features=14, hidden=8, num_layers=1, dropout=0.1)
    cfg = TrainingConfig(
        batch_size=16, epochs=2, lr=1e-3, weight_decay=1e-4, pos_weight=1.0
    )
    history = train_deep_model(m, loader, loader, cfg, device, verbose=False)
    assert len(history.epochs) == 2
    assert history.best_state_dict is not None


def test_ensemble_avg_consistency():
    """Verify simple average ensemble math."""
    from src.models.ensemble import (
        average_classification, average_regression,
    )

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 4.0, 5.0])
    avg = average_regression(a, b)
    assert np.allclose(avg, [2.0, 3.0, 4.0])

    p1 = np.array([0.1, 0.5, 0.9])
    p2 = np.array([0.3, 0.5, 0.7])
    avg_p = average_classification(p1, p2)
    assert np.allclose(avg_p, [0.2, 0.5, 0.8])


def test_phase3_generalization_table_optional():
    p = TABLES_DIR / "phase3_generalization.csv"
    if not p.exists():
        pytest.skip("Run scripts/evaluate_generalization.py first")
    df = pd.read_csv(p)
    assert len(df) > 0
