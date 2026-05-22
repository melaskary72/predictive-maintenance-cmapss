"""Build XGBoost + LSTM + CNN ensemble and evaluate on FD001 test.

Also produces the XGBoost calibration plot here (rather than in evaluate_deep.py)
because unpickling an XGBoost model in the same process as an active MPS torch
context segfaults on Apple Silicon.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.models.ensemble import (
    average_classification,
    average_regression,
)
from src.models.evaluation import classification_metrics, regression_metrics

PRIMARY_HORIZON: int = 30
PHASE2_DIR = PROJECT_ROOT / "results" / "models" / "phase2"
PHASE3_DIR = PROJECT_ROOT / "results" / "models" / "phase3"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "phase3"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_xgboost_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Re-run XGBoost predictions on FD001 test (faster than re-saving Phase 2)."""
    test_df = pd.read_parquet(PROCESSED_DIR / "FD001_tabular_test.parquet")
    with open(PHASE2_DIR / "feature_cols.pkl", "rb") as f:
        feat_cols = pickle.load(f)
    with open(PHASE2_DIR / "reg_xgboost.pkl", "rb") as f:
        reg = pickle.load(f)
    with open(PHASE2_DIR / "clf_xgboost.pkl", "rb") as f:
        clf = pickle.load(f)
    rul_pred = reg.predict(test_df[feat_cols])
    clf_proba = clf.predict_proba(test_df[feat_cols])[:, 1]
    return rul_pred, clf_proba


def main() -> None:
    y_rul = np.load(PHASE3_DIR / "y_rul_test.npy")
    y_clf = np.load(PHASE3_DIR / "y_clf_test.npy")

    lstm_rul = np.load(PHASE3_DIR / "lstm_rul_test.npy")
    lstm_clf = np.load(PHASE3_DIR / "lstm_clf_test.npy")
    cnn_rul = np.load(PHASE3_DIR / "cnn_rul_test.npy")
    cnn_clf = np.load(PHASE3_DIR / "cnn_clf_test.npy")
    xgb_rul, xgb_clf = load_xgboost_predictions()

    n = len(y_rul)
    for name, arr in [("xgb_rul", xgb_rul), ("lstm_rul", lstm_rul),
                       ("cnn_rul", cnn_rul), ("xgb_clf", xgb_clf),
                       ("lstm_clf", lstm_clf), ("cnn_clf", cnn_clf)]:
        assert len(arr) == n, f"{name} has length {len(arr)}, expected {n}"

    rows = []

    avg_rul = average_regression(xgb_rul, lstm_rul, cnn_rul)
    avg_clf = average_classification(xgb_clf, lstm_clf, cnn_clf)
    rows.append({
        "subset": "FD001", "task": "regression",
        "model": "ensemble_avg_xgb_lstm_cnn",
        **regression_metrics(y_rul, avg_rul),
    })
    rows.append({
        "subset": "FD001", "task": f"classification_h{PRIMARY_HORIZON}",
        "model": "ensemble_avg_xgb_lstm_cnn",
        **classification_metrics(y_clf, avg_clf),
    })

    for name, preds_rul, preds_clf in [
        ("ensemble_xgb_lstm", [xgb_rul, lstm_rul], [xgb_clf, lstm_clf]),
        ("ensemble_xgb_cnn", [xgb_rul, cnn_rul], [xgb_clf, cnn_clf]),
        ("ensemble_lstm_cnn", [lstm_rul, cnn_rul], [lstm_clf, cnn_clf]),
    ]:
        rows.append({
            "subset": "FD001", "task": "regression", "model": name,
            **regression_metrics(y_rul, average_regression(*preds_rul)),
        })
        rows.append({
            "subset": "FD001", "task": f"classification_h{PRIMARY_HORIZON}",
            "model": name,
            **classification_metrics(y_clf, average_classification(*preds_clf)),
        })

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "phase3_ensemble_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Ensemble results at {out_path}")
    print(df.to_string(index=False))

    # XGBoost calibration plot (here so it shares the xgboost-loaded process)
    prob_true, prob_pred = calibration_curve(y_clf, xgb_clf, n_bins=10)
    brier = brier_score_loss(y_clf, xgb_clf)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", label="XGBoost")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"XGBoost  |  Brier = {brier:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "xgb_calibration.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Ensemble calibration plot
    prob_true, prob_pred = calibration_curve(y_clf, avg_clf, n_bins=10)
    brier = brier_score_loss(y_clf, avg_clf)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", label="Ensemble (XGB+LSTM+CNN)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Ensemble  |  Brier = {brier:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ensemble_calibration.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Calibration plots written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
