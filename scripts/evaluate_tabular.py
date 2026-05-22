"""Generate diagnostic plots for the Phase 2 tabular models.

Loads saved models from results/models/phase2/ and produces:
- Predicted vs actual RUL scatter plots
- Per-engine RUL trajectories for selected test engines
- ROC and PR curves for classification
- Confusion matrix at threshold 0.5

Run from project root:
    uv run python scripts/evaluate_tabular.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless script

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.models.evaluation import (
    plot_confusion_matrix,
    plot_per_engine_trajectories,
    plot_pred_vs_actual,
    plot_roc_pr_curves,
)
from src.utils.plotting import configure_plot_style

PRIMARY_HORIZON: int = 30
MODELS_DIR: Path = PROJECT_ROOT / "results" / "models" / "phase2"
FIGURES_DIR: Path = PROJECT_ROOT / "results" / "figures" / "phase2"


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    configure_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_parquet(PROCESSED_DIR / "FD001_tabular_test.parquet")
    feat_cols = load_pickle(MODELS_DIR / "feature_cols.pkl")
    X_test = test_df[feat_cols].values
    y_test_rul = test_df["rul_clipped"].values
    y_test_clf = test_df[f"failure_within_{PRIMARY_HORIZON}"].values

    engine_lifetimes = test_df.groupby("unit_id")["last_cycle"].max().sort_values()
    short_engine = int(engine_lifetimes.index[5])
    mid_engine = int(engine_lifetimes.index[len(engine_lifetimes) // 2])
    long_engine = int(engine_lifetimes.index[-5])
    selected = [short_engine, mid_engine, long_engine]
    print(f"Selected engines for trajectory plots: {selected}")

    for name in ["ridge", "random_forest", "xgboost"]:
        model = load_pickle(MODELS_DIR / f"reg_{name}.pkl")
        if name == "xgboost":
            preds = model.predict(test_df[feat_cols])
        else:
            preds = model.predict(X_test)
        plot_pred_vs_actual(
            y_test_rul, preds,
            title=f"Predicted vs Actual RUL — {name} (FD001 test)",
            save_path=FIGURES_DIR / f"reg_{name}_pred_vs_actual.png",
        )
        plot_per_engine_trajectories(
            test_df, preds, selected,
            title=f"Per-engine RUL trajectories — {name}",
            save_path=FIGURES_DIR / f"reg_{name}_trajectories.png",
        )

    for name in ["logreg", "random_forest", "xgboost"]:
        model = load_pickle(MODELS_DIR / f"clf_{name}.pkl")
        if name == "xgboost":
            proba = model.predict_proba(test_df[feat_cols])[:, 1]
        else:
            proba = model.predict_proba(X_test)[:, 1]
        plot_roc_pr_curves(
            y_test_clf, proba,
            title=f"ROC and PR — {name} (failure within {PRIMARY_HORIZON} cycles)",
            save_path=FIGURES_DIR / f"clf_{name}_roc_pr.png",
        )
        y_pred = (proba >= 0.5).astype(int)
        plot_confusion_matrix(
            y_test_clf, y_pred,
            title=f"Confusion matrix — {name} (threshold 0.5)",
            save_path=FIGURES_DIR / f"clf_{name}_confusion.png",
        )

    try:
        import optuna.visualization.matplotlib as ovm
        for kind in ["reg", "clf"]:
            study = load_pickle(MODELS_DIR / f"xgb_{kind}_study.pkl")
            try:
                ovm.plot_param_importances(study)
                import matplotlib.pyplot as plt
                plt.tight_layout()
                plt.savefig(
                    FIGURES_DIR / f"xgb_{kind}_param_importance.png",
                    dpi=120, bbox_inches="tight"
                )
                plt.close()
            except Exception as e:
                print(f"Could not plot importances for {kind}: {e}")
    except ImportError:
        print("optuna.visualization.matplotlib not available, skipping.")

    print(f"\nAll plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
