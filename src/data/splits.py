"""Engine-level train/validation/test splits with no temporal leakage."""
from __future__ import annotations

import numpy as np

from src.config import RANDOM_SEED, VAL_ENGINE_FRACTION


def make_engine_splits(
    train_unit_ids: np.ndarray,
    val_fraction: float = VAL_ENGINE_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Split unique training engines into train and validation engine sets.

    Parameters
    ----------
    train_unit_ids : array of unit IDs from the windowed training data
                     (with repetitions, since each engine has many windows)
    val_fraction : fraction of unique engines to hold out for validation
    seed : random seed for reproducibility

    Returns
    -------
    (train_engine_ids, val_engine_ids) : two arrays of unique engine IDs
    """
    unique_units = np.unique(train_unit_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_units)
    n_val = int(round(len(unique_units) * val_fraction))
    val_engines = np.sort(shuffled[:n_val])
    train_engines = np.sort(shuffled[n_val:])
    return train_engines, val_engines


def mask_by_engines(
    unit_ids: np.ndarray, engine_ids: np.ndarray
) -> np.ndarray:
    """Boolean mask of windows belonging to the given engine set."""
    return np.isin(unit_ids, engine_ids)
