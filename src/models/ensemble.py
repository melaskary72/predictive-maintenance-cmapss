"""Ensemble averaging utilities for combining XGBoost + LSTM + CNN predictions."""
from __future__ import annotations

import numpy as np


def average_regression(*preds: np.ndarray) -> np.ndarray:
    """Simple unweighted average of regression predictions."""
    return np.mean(np.stack(preds, axis=0), axis=0)


def average_classification(*probas: np.ndarray) -> np.ndarray:
    """Simple unweighted average of probability predictions."""
    return np.mean(np.stack(probas, axis=0), axis=0)


def weighted_average_regression(
    preds: list[np.ndarray], weights: list[float]
) -> np.ndarray:
    """Validation-weighted average. Weights are typically 1/RMSE."""
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    return sum(w * p for w, p in zip(weights, preds))


def weighted_average_classification(
    probas: list[np.ndarray], weights: list[float]
) -> np.ndarray:
    """Validation-weighted average. Weights are typically validation AUC."""
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    return sum(w * p for w, p in zip(weights, probas))
