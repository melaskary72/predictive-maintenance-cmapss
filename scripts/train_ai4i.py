"""Train and evaluate XGBoost classifier on AI4I 2020 Predictive Maintenance dataset.

This is cross-dataset generalization: applying our methodology to a structurally
different industrial PM problem to test transferability.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, RANDOM_SEED
from src.data.ai4i import load_ai4i, prepare_ai4i_features
from src.models.evaluation import (
    classification_metrics,
    plot_confusion_matrix,
    plot_roc_pr_curves,
)
from src.models.tabular import make_xgb_classifier
from src.models.tuning import tune_xgb_classifier
from src.utils.plotting import configure_plot_style
from src.utils.seeds import set_seeds

MODELS_DIR = PROJECT_ROOT / "results" / "models" / "phase5_ai4i"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "phase5"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    set_seeds(RANDOM_SEED)
    configure_plot_style()

    print("=" * 70)
    print("Phase 5 Part B: AI4I 2020 Cross-Dataset Generalization")
    print("=" * 70)

    df = load_ai4i()
    print(f"AI4I shape: {df.shape}")
    print(f"Failure rate: {df['failure'].mean():.4f}")
    print(f"\nFailure mode breakdown:")
    for col in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        if col in df.columns:
            print(f"  {col}: {df[col].sum()} ({df[col].mean()*100:.2f}%)")

    X, y, feature_names = prepare_ai4i_features(df)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Feature names: {feature_names}")

    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=RANDOM_SEED
    )
    print(f"\nSplit sizes: train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")
    print(f"Train pos rate: {y_tr.mean():.4f}")
    print(f"Val pos rate:   {y_va.mean():.4f}")
    print(f"Test pos rate:  {y_te.mean():.4f}")

    print("\n--- XGBoost Optuna tuning (25 trials) ---")
    t0 = time.time()
    best_params, study = tune_xgb_classifier(
        X_tr, y_tr, X_va, y_va, n_trials=25, seed=RANDOM_SEED
    )
    print(f"Best params: {best_params}")
    print(f"Best val ROC AUC: {study.best_value:.4f}")
    tuning_seconds = time.time() - t0

    model = make_xgb_classifier(**best_params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    test_proba = model.predict_proba(X_te)[:, 1]
    val_proba = model.predict_proba(X_va)[:, 1]

    test_metrics = classification_metrics(y_te, test_proba)
    val_metrics = classification_metrics(y_va, val_proba)

    print(f"\nFinal test ROC AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Final test PR AUC:  {test_metrics['pr_auc']:.4f}")
    print(f"Final test F1:      {test_metrics['f1']:.3f}")
    print(f"Final test Brier:   {test_metrics['brier']:.4f}")

    print("\n--- 5-fold cross-validation (using best params) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_aucs, cv_pr_aucs, cv_f1s = [], [], []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        m = make_xgb_classifier(**best_params)
        m.fit(X[tr_idx], y[tr_idx])
        proba = m.predict_proba(X[te_idx])[:, 1]
        metrics = classification_metrics(y[te_idx], proba)
        cv_aucs.append(metrics["roc_auc"])
        cv_pr_aucs.append(metrics["pr_auc"])
        cv_f1s.append(metrics["f1"])
        print(f"  Fold {fold}: AUC {metrics['roc_auc']:.4f}, "
              f"PR-AUC {metrics['pr_auc']:.4f}, F1 {metrics['f1']:.3f}")
    print(f"\n5-fold CV ROC AUC: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")
    print(f"5-fold CV PR AUC:  {np.mean(cv_pr_aucs):.4f} +/- {np.std(cv_pr_aucs):.4f}")
    print(f"5-fold CV F1:      {np.mean(cv_f1s):.3f} +/- {np.std(cv_f1s):.3f}")

    with open(MODELS_DIR / "xgb_ai4i.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / "xgb_ai4i_study.pkl", "wb") as f:
        pickle.dump(study, f)
    with open(MODELS_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    np.save(MODELS_DIR / "feature_names.npy", np.array(feature_names))
    np.save(MODELS_DIR / "test_proba.npy", test_proba)

    plot_roc_pr_curves(
        y_te, test_proba, "AI4I 2020 - XGBoost ROC and PR curves",
        save_path=FIGURES_DIR / "ai4i_roc_pr.png",
    )
    plot_confusion_matrix(
        y_te, (test_proba >= 0.5).astype(int),
        "AI4I 2020 - XGBoost Confusion (threshold 0.5)",
        save_path=FIGURES_DIR / "ai4i_confusion.png",
    )

    importance = model.feature_importances_
    order = np.argsort(importance)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        np.array(feature_names)[order][::-1],
        importance[order][::-1],
        color="#1f77b4",
    )
    ax.set_xlabel("XGBoost feature importance (gain)")
    ax.set_title("AI4I 2020 - XGBoost feature importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ai4i_feature_importance.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)

    rows = [{
        "dataset": "ai4i_2020",
        "model": "xgb_optuna",
        "split": "test",
        **test_metrics,
        "train_seconds": tuning_seconds,
        "n_train": len(X_tr),
        "n_val": len(X_va),
        "n_test": len(X_te),
    }]
    rows.append({
        "dataset": "ai4i_2020",
        "model": "xgb_optuna_5fold_cv",
        "split": "cv5",
        "roc_auc": float(np.mean(cv_aucs)),
        "roc_auc_std": float(np.std(cv_aucs)),
        "pr_auc": float(np.mean(cv_pr_aucs)),
        "pr_auc_std": float(np.std(cv_pr_aucs)),
        "f1": float(np.mean(cv_f1s)),
        "f1_std": float(np.std(cv_f1s)),
        "n_total": len(X),
    })

    df_results = pd.DataFrame(rows)
    out_path = TABLES_DIR / "ai4i_results.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
