"""Central configuration for the predictive maintenance project.

All paths, hyperparameters, and constants live here. Every other module
imports from this file to ensure consistency across phases.
"""
from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
CMAPSS_RAW_DIR: Path = RAW_DIR / "cmapss"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
RANDOM_SEED: int = 42

# -----------------------------------------------------------------------------
# C-MAPSS dataset constants
# -----------------------------------------------------------------------------
CMAPSS_SUBSETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")

# All raw column names in C-MAPSS files (in order)
CMAPSS_RAW_COLUMNS: list[str] = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Operational setting columns
OP_SETTING_COLS: list[str] = ["op_setting_1", "op_setting_2", "op_setting_3"]

# All sensor columns
ALL_SENSOR_COLS: list[str] = [f"sensor_{i}" for i in range(1, 22)]

# Variance threshold for dropping flat sensors (computed empirically per subset)
SENSOR_VARIANCE_THRESHOLD: float = 1e-4

# Number of operating regime clusters for FD002 and FD004
N_OP_REGIMES: int = 6

# -----------------------------------------------------------------------------
# Labeling
# -----------------------------------------------------------------------------
# Piecewise-linear RUL ceiling. Standard practice in C-MAPSS literature.
RUL_CEILING: int = 125

# Prediction horizons in cycles for binary classification
PREDICTION_HORIZONS: tuple[int, ...] = (10, 30, 50)

# -----------------------------------------------------------------------------
# Sequence construction
# -----------------------------------------------------------------------------
WINDOW_SIZE: int = 30
WINDOW_STRIDE: int = 1

# -----------------------------------------------------------------------------
# Splits
# -----------------------------------------------------------------------------
# Fraction of training engines held out as validation
VAL_ENGINE_FRACTION: float = 0.2
