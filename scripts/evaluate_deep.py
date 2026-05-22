"""Diagnostic plots, calibration analysis, and SHAP for Phase 3 deep models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.models.deep import DilatedConv1DRegressor, LSTMRegressor
from src.models.evaluation import (
    plot_confusion_matrix,
    plot_per_engine_trajectories,
    plot_pred_vs_actual,
    plot_roc_pr_curves,
)
from src.utils.plotting import configure_plot_style
from src.utils.seeds import set_seeds

PRIMARY_HORIZON: int = 30
PHASE3_DIR = PROJECT_ROOT / "results" / "models" / "phase3"
PHASE2_DIR = PROJECT_ROOT / "results" / "models" / "phase2"
FIGURES_DIR: Path = PROJECT_ROOT / "results" / "figures" / "phase3"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_calibration(
    y_true: np.ndarray, y_proba: np.ndarray, title: str, save_path: Path
) -> None:
    """Reliability diagram with Brier score annotation."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    brier = brier_score_loss(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", label=f"{title}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{title}  |  Brier = {brier:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seeds(42)
    configure_plot_style()
    device = get_device()
    print(f"Device: {device}")

    test = np.load(PROCESSED_DIR / "FD001_windows_test.npz")
    X_test = test["X"].astype(np.float32)
    y_rul = test["y_rul"]
    y_clf = test[f"y_class_h{PRIMARY_HORIZON}"]

    with open(PHASE3_DIR / "best_params.json") as f:
        best = json.load(f)

    # ---- LSTM diagnostics ----
    lstm = LSTMRegressor(
        n_features=X_test.shape[-1],
        hidden=best["lstm"]["hidden"],
        num_layers=best["lstm"]["num_layers"],
        dropout=best["lstm"]["dropout"],
    ).to(device)
    ckpt = torch.load(PHASE3_DIR / "lstm_final.pt", map_location=device,
                       weights_only=False)
    lstm.load_state_dict(ckpt["state_dict"])

    lstm_rul = np.load(PHASE3_DIR / "lstm_rul_test.npy")
    lstm_clf = np.load(PHASE3_DIR / "lstm_clf_test.npy")

    plot_pred_vs_actual(
        y_rul, lstm_rul, "LSTM — Predicted vs Actual RUL (FD001)",
        save_path=FIGURES_DIR / "lstm_pred_vs_actual.png",
    )
    plot_roc_pr_curves(
        y_clf, lstm_clf, f"LSTM — Failure within {PRIMARY_HORIZON} cycles",
        save_path=FIGURES_DIR / "lstm_roc_pr.png",
    )
    plot_confusion_matrix(
        y_clf, (lstm_clf >= 0.5).astype(int),
        "LSTM — Confusion (threshold 0.5)",
        save_path=FIGURES_DIR / "lstm_confusion.png",
    )
    plot_calibration(
        y_clf, lstm_clf, "LSTM",
        save_path=FIGURES_DIR / "lstm_calibration.png",
    )

    # ---- CNN diagnostics ----
    cnn = DilatedConv1DRegressor(
        n_features=X_test.shape[-1],
        n_channels=best["cnn"]["n_channels"],
        n_blocks=best["cnn"]["n_blocks"],
        kernel_size=best["cnn"]["kernel_size"],
        dropout=best["cnn"]["dropout"],
    ).to(device)
    ckpt = torch.load(PHASE3_DIR / "cnn_final.pt", map_location=device,
                       weights_only=False)
    cnn.load_state_dict(ckpt["state_dict"])

    cnn_rul = np.load(PHASE3_DIR / "cnn_rul_test.npy")
    cnn_clf = np.load(PHASE3_DIR / "cnn_clf_test.npy")

    plot_pred_vs_actual(
        y_rul, cnn_rul, "CNN — Predicted vs Actual RUL (FD001)",
        save_path=FIGURES_DIR / "cnn_pred_vs_actual.png",
    )
    plot_roc_pr_curves(
        y_clf, cnn_clf, f"CNN — Failure within {PRIMARY_HORIZON} cycles",
        save_path=FIGURES_DIR / "cnn_roc_pr.png",
    )
    plot_confusion_matrix(
        y_clf, (cnn_clf >= 0.5).astype(int),
        "CNN — Confusion (threshold 0.5)",
        save_path=FIGURES_DIR / "cnn_confusion.png",
    )
    plot_calibration(
        y_clf, cnn_clf, "CNN",
        save_path=FIGURES_DIR / "cnn_calibration.png",
    )

    # ---- XGBoost calibration runs in run_ensemble.py to avoid an MPS+xgboost
    # unpickle segfault when both are loaded in the same process.

    # ---- Trajectory plots ----
    test_tab = pd.read_parquet(PROCESSED_DIR / "FD001_tabular_test.parquet")
    engine_lifetimes = test_tab.groupby("unit_id")["last_cycle"].max().sort_values()
    short = int(engine_lifetimes.index[5])
    mid = int(engine_lifetimes.index[len(engine_lifetimes) // 2])
    long = int(engine_lifetimes.index[-5])
    selected = [short, mid, long]

    plot_per_engine_trajectories(
        test_tab, lstm_rul, selected,
        title="LSTM — Per-engine RUL trajectories",
        save_path=FIGURES_DIR / "lstm_trajectories.png",
    )
    plot_per_engine_trajectories(
        test_tab, cnn_rul, selected,
        title="CNN — Per-engine RUL trajectories",
        save_path=FIGURES_DIR / "cnn_trajectories.png",
    )

    # ---- SHAP on LSTM ----
    print("Running SHAP on LSTM...")
    try:
        import shap

        n_explain = 100
        rng = np.random.default_rng(42)
        idx = rng.choice(X_test.shape[0], size=n_explain, replace=False)
        background_idx = rng.choice(X_test.shape[0], size=50, replace=False)

        background = torch.from_numpy(X_test[background_idx]).to(device)
        sample = torch.from_numpy(X_test[idx]).to(device)

        class RegOnly(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, x):
                reg, _ = self.base(x)
                return reg.unsqueeze(-1)

        wrapped = RegOnly(lstm).to(device).eval()

        explainer = shap.GradientExplainer(wrapped, background)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values)
        # GradientExplainer returns (batch, seq, features, n_outputs);
        # collapse the singleton output axis if present.
        if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
            shap_values = shap_values[..., 0]

        feat_importance = np.abs(shap_values).mean(axis=(0, 1))

        sensor_cols = test["sensor_cols"]
        order = np.argsort(feat_importance)[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            np.array(sensor_cols)[order][::-1],
            feat_importance[order][::-1],
        )
        ax.set_xlabel("Mean |SHAP value| (RUL prediction)")
        ax.set_title("LSTM feature importance via SHAP GradientExplainer")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "lstm_shap_importance.png", dpi=120,
                    bbox_inches="tight")
        plt.close(fig)
        np.save(PHASE3_DIR / "lstm_shap_values.npy", shap_values)
        print("SHAP done")
    except Exception as exc:
        print(f"SHAP failed (non-fatal): {exc}")

    # ---- Training curves ----
    for name in ["lstm_final", "cnn_final", "lstm_aux_final", "cnn_aux_final"]:
        try:
            ckpt = torch.load(PHASE3_DIR / f"{name}.pt", map_location="cpu",
                               weights_only=False)
            history = ckpt["history"]
            train_losses = [h["train_loss"] for h in history]
            val_losses = [h["val_loss"] for h in history]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(train_losses, label="Train")
            ax.plot(val_losses, label="Val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Total loss")
            ax.set_title(f"{name} training curves")
            ax.legend()
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"{name}_training_curves.png",
                        dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"Could not plot {name} curves: {exc}")

    print(f"\nAll plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
