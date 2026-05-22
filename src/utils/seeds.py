"""Reproducible seeding for all stochastic libraries used in this project."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seeds(seed: int = 42) -> None:
    """Set seeds for Python, numpy, and torch (if available).

    Call this at the top of every script and notebook before any
    stochastic operation.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass
