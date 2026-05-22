"""Shared matplotlib styling for the project."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def configure_plot_style() -> None:
    """Apply consistent styling across all plots in the project."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
