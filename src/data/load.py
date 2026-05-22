"""Loaders for raw C-MAPSS text files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CMAPSS_RAW_COLUMNS, CMAPSS_RAW_DIR


def load_cmapss_train(subset: str) -> pd.DataFrame:
    """Load a C-MAPSS training subset (e.g. 'FD001').

    Each row is one cycle of one engine. Engines run to failure.

    Returns
    -------
    DataFrame with columns from CMAPSS_RAW_COLUMNS plus dtype-correct values.
    """
    path = CMAPSS_RAW_DIR / f"train_{subset}.txt"
    return _load_raw_file(path)


def load_cmapss_test(subset: str) -> pd.DataFrame:
    """Load a C-MAPSS test subset. Engines are truncated before failure."""
    path = CMAPSS_RAW_DIR / f"test_{subset}.txt"
    return _load_raw_file(path)


def load_cmapss_rul(subset: str) -> np.ndarray:
    """Load the RUL ground truth for the test subset.

    Returns a 1D array where index i holds the true RUL of test engine (i+1)
    at the engine's last observed cycle.
    """
    path = CMAPSS_RAW_DIR / f"RUL_{subset}.txt"
    return pd.read_csv(path, header=None, sep=r"\s+").values.flatten()


def _load_raw_file(path: Path) -> pd.DataFrame:
    """C-MAPSS raw text files are whitespace-separated, no header.

    The format has 26 columns: unit, cycle, 3 op settings, 21 sensors.
    Some files have trailing whitespace producing extra empty columns,
    which we drop.
    """
    df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
    # Some files produce 26 valid columns plus trailing NaN columns from
    # extra whitespace. Keep only the first 26.
    df = df.iloc[:, : len(CMAPSS_RAW_COLUMNS)]
    df.columns = CMAPSS_RAW_COLUMNS
    df["unit"] = df["unit"].astype(np.int32)
    df["cycle"] = df["cycle"].astype(np.int32)
    return df
