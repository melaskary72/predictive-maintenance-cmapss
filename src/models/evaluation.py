"""Shared evaluation utilities: metrics, plotting, NASA scoring function."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA C-MAPSS asymmetric scoring function.

    Penalizes late predictions (predicting RUL > true) more heavily than
    early predictions, because late predictions cause unscheduled failures.

    Score = sum over samples of:
        exp(-(y_pred - y_true) / 13) - 1   if y_pred < y_true
        exp((y_pred - y_true) / 10) - 1    if y_pred >= y_true
    """
    diff = y_pred - y_true
    score = np.where(
        diff < 0,
        np.exp(-diff / 13) - 1,
        np.exp(diff / 10) - 1,
    )
    return float(score.sum())


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics for RUL prediction."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "nasa_score": nasa_score(y_true, y_pred),
        "n_samples": int(len(y_true)),
    }


def classification_metrics(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> dict:
    """Compute classification metrics from predicted probabilities."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    pos_rate = float(y_true.mean())
    if pos_rate == 0.0 or pos_rate == 1.0:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "brier": brier_score_loss(y_true, y_pred_proba),
            "positive_rate": pos_rate,
            "n_samples": int(len(y_true)),
        }
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_pred_proba)),
        "positive_rate": pos_rate,
        "n_samples": int(len(y_true)),
    }


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path | None = None,
) -> None:
    """Scatter plot of predicted vs actual RUL with y=x reference line."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.3, s=8)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_engine_trajectories(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    engine_ids: list[int],
    title: str,
    save_path: Path | None = None,
) -> None:
    """Plot predicted vs actual RUL trajectories over time for selected engines.

    df_test must contain columns: 'unit_id', 'last_cycle', 'rul_clipped'.
    y_pred is aligned to df_test row order.
    """
    df = df_test.copy()
    df["y_pred"] = y_pred
    n = len(engine_ids)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, engine_id in zip(axes, engine_ids):
        sub = df[df["unit_id"] == engine_id].sort_values("last_cycle")
        ax.plot(sub["last_cycle"], sub["rul_clipped"], "o-",
                label="Actual", markersize=3)
        ax.plot(sub["last_cycle"], sub["y_pred"], "s-",
                label="Predicted", markersize=3, alpha=0.8)
        ax.set_xlabel("Cycle")
        ax.set_title(f"Engine {engine_id}")
        ax.legend()
    axes[0].set_ylabel("RUL (cycles)")
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    title: str,
    save_path: Path | None = None,
) -> None:
    """Plot ROC and Precision-Recall curves side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    ax1.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    ax2.plot(recall, precision, label=f"AP = {pr_auc:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path | None = None,
) -> None:
    """Plot a 2x2 confusion matrix with annotated counts and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No failure", "Failure"])
    ax.set_yticklabels(["No failure", "Failure"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = 100 * count / total
            color = "white" if count > total / 4 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", color=color)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
