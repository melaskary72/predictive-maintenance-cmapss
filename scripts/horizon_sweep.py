"""Run all classifiers across horizons 10, 30, 50 on FD001.

For XGBoost: retrain freshly at each horizon with light Optuna search (25 trials).
For deep models: reuse existing predictions, compute metrics at alternate horizons.

Saves results to results/tables/horizon_sweep.csv and produces a horizon-vs-AUC
comparison plot.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    PREDICTION_HORIZONS,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from src.models.evaluation import classification_metrics
from src.models.tabular import make_xgb_classifier
from src.models.tuning import tune_xgb_classifier
from src.utils.plotting import configure_plot_style
from src.utils.seeds import set_seeds

PHASE2_DIR = PROJECT_ROOT / "results" / "models" / "phase2"
PHASE3_DIR = PROJECT_ROOT / "results" / "models" / "phase3"
PHASE4_DIR = PROJECT_ROOT / "results" / "models" / "phase4"
PHASE5_DIR = PROJECT_ROOT / "results" / "models" / "phase5"
PHASE5_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "phase5"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_tabular_arrays():
    """Load FD001 tabular train/val/test with all horizon labels."""
    tr_df = pd.read_parquet(PROCESSED_DIR / "FD001_tabular_train.parquet")
    te_df = pd.read_parquet(PROCESSED_DIR / "FD001_tabular_test.parquet")
    windows = np.load(PROCESSED_DIR / "FD001_windows_train.npz")
    train_engines = set(windows["train_engine_ids"].tolist())
    val_engines = set(windows["val_engine_ids"].tolist())
    train_mask = tr_df["unit_id"].isin(train_engines).values
    val_mask = tr_df["unit_id"].isin(val_engines).values
    drop_cols = {"unit_id", "last_cycle", "rul_clipped"}
    drop_cols |= {f"failure_within_{h}" for h in PREDICTION_HORIZONS}
    feat_cols = [c for c in tr_df.columns if c not in drop_cols]
    return tr_df, te_df, train_mask, val_mask, feat_cols


def load_window_labels_test(horizon: int) -> np.ndarray:
    """Load test set classification labels for a given horizon (window-based)."""
    arr = np.load(PROCESSED_DIR / "FD001_windows_test.npz")
    return arr[f"y_class_h{horizon}"]


def train_xgb_at_horizon(
    horizon: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    feat_cols: list[str],
    n_trials: int = 25,
) -> tuple[dict, np.ndarray, dict]:
    """Train fresh XGBoost classifier at the given horizon."""
    target = f"failure_within_{horizon}"
    X_tr = train_df[train_mask][feat_cols].values
    y_tr = train_df[train_mask][target].values
    X_va = train_df[val_mask][feat_cols].values
    y_va = train_df[val_mask][target].values
    X_te = test_df[feat_cols].values
    y_te = test_df[target].values

    pos_train = float(y_tr.sum())
    pos_test = float(y_te.sum())
    print(f"  Horizon {horizon}: pos_train={pos_train/len(y_tr):.3f}, "
          f"pos_test={pos_test/len(y_te):.3f}")

    if pos_train < 5 or (len(y_tr) - pos_train) < 5:
        print(f"  Skipping h={horizon}: insufficient positives in train")
        return {}, np.zeros(len(y_te)), {}

    best_params, study = tune_xgb_classifier(
        X_tr, y_tr, X_va, y_va, n_trials=n_trials, seed=RANDOM_SEED
    )
    model = make_xgb_classifier(**best_params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = classification_metrics(y_te, proba)
    with open(PHASE5_DIR / f"xgb_h{horizon}.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(PHASE5_DIR / f"xgb_h{horizon}_study.pkl", "wb") as f:
        pickle.dump(study, f)
    np.save(PHASE5_DIR / f"xgb_h{horizon}_proba.npy", proba)
    return best_params, proba, metrics


def evaluate_deep_at_alternate_horizons() -> list[dict]:
    """For LSTM, CNN, Transformer, reuse existing test predictions and compute
    metrics at horizons 10 and 50.

    Note: these deep models were TRAINED with h=30 as their classification target.
    Their predicted probabilities are calibrated for that horizon. Evaluating them
    at h=10 and h=50 tests whether the learned representations transfer.
    """
    rows = []
    deep_predictions = {
        "lstm": PHASE3_DIR / "lstm_clf_test.npy",
        "cnn": PHASE3_DIR / "cnn_clf_test.npy",
        "transformer": PHASE4_DIR / "transformer_clf_test.npy",
    }
    for h in PREDICTION_HORIZONS:
        y_h = load_window_labels_test(h)
        for name, path in deep_predictions.items():
            if not path.exists():
                continue
            proba = np.load(path)
            metrics = classification_metrics(y_h, proba)
            rows.append({
                "horizon": h,
                "model": f"{name}_at_h30_eval_at_h{h}",
                "model_short": name,
                "training_horizon": 30,
                "eval_horizon": h,
                **metrics,
            })
    return rows


def main() -> None:
    set_seeds(RANDOM_SEED)
    configure_plot_style()
    print("=" * 70)
    print("Phase 5 Part A: Horizon Sweep")
    print("=" * 70)

    tr_df, te_df, train_mask, val_mask, feat_cols = load_tabular_arrays()
    rows = []
    xgb_probas = {}

    print("\n--- XGBoost at each horizon (fresh training) ---")
    for h in PREDICTION_HORIZONS:
        print(f"\nHorizon {h}:")
        t0 = time.time()
        best_params, proba, metrics = train_xgb_at_horizon(
            h, tr_df, te_df, train_mask, val_mask, feat_cols, n_trials=25,
        )
        if metrics:
            xgb_probas[h] = proba
            rows.append({
                "horizon": h,
                "model": f"xgb_at_h{h}",
                "model_short": "xgb",
                "training_horizon": h,
                "eval_horizon": h,
                **metrics,
                "train_seconds": time.time() - t0,
            })
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}, "
                  f"PR AUC: {metrics['pr_auc']:.4f}, "
                  f"F1: {metrics['f1']:.3f}")

    print("\n--- Deep models evaluated at all horizons (no retraining) ---")
    deep_rows = evaluate_deep_at_alternate_horizons()
    rows.extend(deep_rows)
    for r in deep_rows:
        print(f"  {r['model']}: ROC AUC {r['roc_auc']:.4f}")

    print("\n--- Ensembles at each horizon ---")
    for h in PREDICTION_HORIZONS:
        if h not in xgb_probas:
            continue
        y_h = load_window_labels_test(h)

        for deep_name, deep_path in [
            ("lstm", PHASE3_DIR / "lstm_clf_test.npy"),
            ("cnn", PHASE3_DIR / "cnn_clf_test.npy"),
            ("transformer", PHASE4_DIR / "transformer_clf_test.npy"),
        ]:
            if not deep_path.exists():
                continue
            deep_proba = np.load(deep_path)
            if len(deep_proba) != len(xgb_probas[h]):
                print(f"  Skipping ensemble for h={h} ({deep_name}): length mismatch")
                continue
            ens_proba = 0.5 * (xgb_probas[h] + deep_proba)
            metrics = classification_metrics(y_h, ens_proba)
            rows.append({
                "horizon": h,
                "model": f"ensemble_xgb_h{h}_plus_{deep_name}_at_h30",
                "model_short": f"ensemble_xgb_{deep_name}",
                "training_horizon": h,
                "eval_horizon": h,
                **metrics,
            })

        deep_arrays = []
        for deep_path in [
            PHASE3_DIR / "lstm_clf_test.npy",
            PHASE3_DIR / "cnn_clf_test.npy",
            PHASE4_DIR / "transformer_clf_test.npy",
        ]:
            if deep_path.exists():
                deep_arrays.append(np.load(deep_path))
        if len(deep_arrays) == 3 and all(len(a) == len(xgb_probas[h]) for a in deep_arrays):
            avg = np.mean(np.stack([xgb_probas[h]] + deep_arrays, axis=0), axis=0)
            metrics = classification_metrics(y_h, avg)
            rows.append({
                "horizon": h,
                "model": f"ensemble_4way_at_h{h}",
                "model_short": "ensemble_4way",
                "training_horizon": h,
                "eval_horizon": h,
                **metrics,
            })

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "horizon_sweep.csv"
    df.to_csv(out_path, index=False)
    print(f"\nHorizon sweep results saved to {out_path}")

    fig, ax = plt.subplots(figsize=(9, 6))
    plot_models = ["xgb", "lstm", "cnn", "transformer", "ensemble_4way"]
    colors = {"xgb": "#1f77b4", "lstm": "#ff7f0e", "cnn": "#2ca02c",
              "transformer": "#d62728", "ensemble_4way": "#9467bd"}
    for short_name in plot_models:
        sub = df[df["model_short"] == short_name].sort_values("horizon")
        if len(sub) == 0:
            continue
        ax.plot(sub["horizon"], sub["roc_auc"], "o-",
                label=short_name, color=colors.get(short_name), linewidth=2)
    ax.set_xlabel("Prediction horizon (cycles before failure)")
    ax.set_ylabel("Test ROC AUC")
    ax.set_title("Classification ROC AUC vs prediction horizon (FD001)")
    ax.set_xticks(list(PREDICTION_HORIZONS))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "horizon_roc_auc.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for short_name in plot_models:
        sub = df[df["model_short"] == short_name].sort_values("horizon")
        if len(sub) == 0:
            continue
        ax.plot(sub["horizon"], sub["pr_auc"], "o-",
                label=short_name, color=colors.get(short_name), linewidth=2)
    ax.set_xlabel("Prediction horizon (cycles before failure)")
    ax.set_ylabel("Test PR AUC")
    ax.set_title("Classification PR AUC vs prediction horizon (FD001)")
    ax.set_xticks(list(PREDICTION_HORIZONS))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "horizon_pr_auc.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    pos_rates = []
    for h in PREDICTION_HORIZONS:
        y_h = load_window_labels_test(h)
        pos_rates.append(float(y_h.mean()))
    ax.bar(list(PREDICTION_HORIZONS), pos_rates,
           color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_xlabel("Horizon (cycles)")
    ax.set_ylabel("Test set positive class rate")
    ax.set_title("Class imbalance by horizon (FD001 test)")
    for h, r in zip(PREDICTION_HORIZONS, pos_rates):
        ax.text(h, r + 0.005, f"{r:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "horizon_positive_rate.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
