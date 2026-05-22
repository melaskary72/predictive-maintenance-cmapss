"""Build full 4-model ensemble (XGB + LSTM + CNN + Transformer) and master comparison."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.models.ensemble import (
    average_classification,
    average_regression,
)
from src.models.evaluation import classification_metrics, regression_metrics

PRIMARY_HORIZON = 30
PHASE2_DIR = PROJECT_ROOT / "results" / "models" / "phase2"
PHASE3_DIR = PROJECT_ROOT / "results" / "models" / "phase3"
PHASE4_DIR = PROJECT_ROOT / "results" / "models" / "phase4"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def load_xgb_predictions():
    test_df = pd.read_parquet(PROCESSED_DIR / "FD001_tabular_test.parquet")
    with open(PHASE2_DIR / "feature_cols.pkl", "rb") as f:
        feat_cols = pickle.load(f)
    with open(PHASE2_DIR / "reg_xgboost.pkl", "rb") as f:
        reg = pickle.load(f)
    with open(PHASE2_DIR / "clf_xgboost.pkl", "rb") as f:
        clf = pickle.load(f)
    return reg.predict(test_df[feat_cols]), clf.predict_proba(test_df[feat_cols])[:, 1]


def main() -> None:
    y_rul = np.load(PHASE3_DIR / "y_rul_test.npy")
    y_clf = np.load(PHASE3_DIR / "y_clf_test.npy")

    lstm_rul = np.load(PHASE3_DIR / "lstm_rul_test.npy")
    lstm_clf = np.load(PHASE3_DIR / "lstm_clf_test.npy")
    cnn_rul = np.load(PHASE3_DIR / "cnn_rul_test.npy")
    cnn_clf = np.load(PHASE3_DIR / "cnn_clf_test.npy")
    tx_rul = np.load(PHASE4_DIR / "transformer_rul_test.npy")
    tx_clf = np.load(PHASE4_DIR / "transformer_clf_test.npy")
    xgb_rul, xgb_clf = load_xgb_predictions()

    n = len(y_rul)
    for name, arr in [
        ("xgb_rul", xgb_rul), ("lstm_rul", lstm_rul),
        ("cnn_rul", cnn_rul), ("tx_rul", tx_rul),
        ("xgb_clf", xgb_clf), ("lstm_clf", lstm_clf),
        ("cnn_clf", cnn_clf), ("tx_clf", tx_clf),
    ]:
        assert len(arr) == n, f"{name} length mismatch: {len(arr)} vs {n}"

    rows = []

    # 4-way average
    rul = average_regression(xgb_rul, lstm_rul, cnn_rul, tx_rul)
    clf = average_classification(xgb_clf, lstm_clf, cnn_clf, tx_clf)
    rows.append({
        "subset": "FD001", "task": "regression",
        "model": "ensemble_avg_xgb_lstm_cnn_tx",
        **regression_metrics(y_rul, rul),
    })
    rows.append({
        "subset": "FD001", "task": f"classification_h{PRIMARY_HORIZON}",
        "model": "ensemble_avg_xgb_lstm_cnn_tx",
        **classification_metrics(y_clf, clf),
    })

    pairs = {
        "ensemble_xgb_tx": ([xgb_rul, tx_rul], [xgb_clf, tx_clf]),
        "ensemble_lstm_tx": ([lstm_rul, tx_rul], [lstm_clf, tx_clf]),
        "ensemble_cnn_tx": ([cnn_rul, tx_rul], [cnn_clf, tx_clf]),
        "ensemble_xgb_lstm_tx": (
            [xgb_rul, lstm_rul, tx_rul], [xgb_clf, lstm_clf, tx_clf]
        ),
        "ensemble_xgb_cnn_tx": (
            [xgb_rul, cnn_rul, tx_rul], [xgb_clf, cnn_clf, tx_clf]
        ),
        "ensemble_lstm_cnn_tx": (
            [lstm_rul, cnn_rul, tx_rul], [lstm_clf, cnn_clf, tx_clf]
        ),
    }
    for name, (rul_list, clf_list) in pairs.items():
        rows.append({
            "subset": "FD001", "task": "regression", "model": name,
            **regression_metrics(y_rul, average_regression(*rul_list)),
        })
        rows.append({
            "subset": "FD001", "task": f"classification_h{PRIMARY_HORIZON}",
            "model": name,
            **classification_metrics(y_clf, average_classification(*clf_list)),
        })

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "phase4_ensemble_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Phase 4 ensemble results at {out_path}")
    print(df.to_string(index=False))

    # ---- Master comparison table ----
    print("\n" + "=" * 70)
    print("Building master comparison table")
    print("=" * 70)

    p2 = pd.read_csv(TABLES_DIR / "phase2_results.csv")
    p3 = pd.read_csv(TABLES_DIR / "phase3_results.csv")
    p3_ens = pd.read_csv(TABLES_DIR / "phase3_ensemble_results.csv")
    p4 = pd.read_csv(TABLES_DIR / "phase4_results.csv")
    p4_ens = df

    def fd001_only(d):
        return d[d["subset"] == "FD001"].copy()

    common_cols = ["model", "task", "rmse", "roc_auc", "pr_auc", "f1", "brier"]
    individuals = pd.concat([
        fd001_only(p2)[common_cols],
        fd001_only(p3)[common_cols],
        fd001_only(p4)[common_cols],
    ], ignore_index=True)

    ensembles = pd.concat([
        fd001_only(p3_ens)[common_cols],
        fd001_only(p4_ens)[common_cols],
    ], ignore_index=True)

    master = pd.concat([individuals, ensembles], ignore_index=True)
    master_path = TABLES_DIR / "master_comparison.csv"
    master.to_csv(master_path, index=False)
    print(f"\nMaster comparison saved to {master_path}")

    for task in master["task"].unique():
        if pd.isna(task):
            continue
        print(f"\n--- Best models on {task} ---")
        sub = master[master["task"] == task]
        if "regression" in str(task):
            print(sub.sort_values("rmse").head(8).to_string(index=False))
        else:
            print(sub.sort_values("roc_auc", ascending=False).head(8)
                  .to_string(index=False))


if __name__ == "__main__":
    main()
