"""Build notebooks/01_eda.ipynb programmatically with nbformat.

Run from project root:
    uv run python scripts/build_eda_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def build() -> None:
    nb = nbf.v4.new_notebook()
    cells: list = []

    cells.append(nbf.v4.new_markdown_cell(
        "# Exploratory Data Analysis: NASA C-MAPSS Turbofan Engine Degradation\n\n"
        "This notebook documents the dataset structure, motivates the preprocessing choices in "
        "`scripts/prepare_data.py`, and produces the eight required visualizations for Phase 1.\n\n"
        "All figures are also saved to `results/eda/` as PNGs."
    ))

    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n\n"
        "# Add project root to path\n"
        "sys.path.insert(0, str(Path.cwd().parent))\n\n"
        "import json\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "from src.config import (\n"
        "    ALL_SENSOR_COLS,\n"
        "    CMAPSS_SUBSETS,\n"
        "    INTERIM_DIR,\n"
        "    OP_SETTING_COLS,\n"
        "    PROCESSED_DIR,\n"
        "    PREDICTION_HORIZONS,\n"
        "    RESULTS_DIR,\n"
        "    RUL_CEILING,\n"
        ")\n"
        "from src.data.load import load_cmapss_train\n"
        "from src.data.preprocess import add_rul_to_train\n"
        "from src.utils.plotting import configure_plot_style\n"
        "from src.utils.seeds import set_seeds\n\n"
        "set_seeds(42)\n"
        "configure_plot_style()\n\n"
        "EDA_DIR = RESULTS_DIR / 'eda'\n"
        "EDA_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "with open(PROCESSED_DIR / 'manifest.json') as f:\n"
        "    manifest = json.load(f)\n"
        "manifest_df = pd.DataFrame(manifest)\n"
        "print(f'Loaded manifest with {len(manifest_df)} subsets')"
    ))

    # ----- 1. Dataset overview table -----
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Dataset overview\n\n"
        "FD001 and FD003 have one operating condition; FD002 and FD004 have six. "
        "FD001 has the simplest fault mode and is our primary benchmark."
    ))
    cells.append(nbf.v4.new_code_cell(
        "rows = []\n"
        "for subset in CMAPSS_SUBSETS:\n"
        "    train = pd.read_parquet(INTERIM_DIR / f'{subset}_train.parquet')\n"
        "    test = pd.read_parquet(INTERIM_DIR / f'{subset}_test.parquet')\n"
        "    sub_manifest = next(m for m in manifest if m['subset'] == subset)\n"
        "    rows.append({\n"
        "        'subset': subset,\n"
        "        'train_engines': train['unit'].nunique(),\n"
        "        'train_cycles': len(train),\n"
        "        'test_engines': test['unit'].nunique(),\n"
        "        'test_cycles': len(test),\n"
        "        'sensors_kept': sub_manifest['n_features'],\n"
        "        'flat_dropped': len(sub_manifest['flat_sensors_dropped']),\n"
        "    })\n"
        "overview = pd.DataFrame(rows).set_index('subset')\n"
        "print(overview.to_string())\n\n"
        "# Render as a saved figure too\n"
        "fig, ax = plt.subplots(figsize=(10, 2.5))\n"
        "ax.axis('off')\n"
        "table = ax.table(\n"
        "    cellText=overview.reset_index().values,\n"
        "    colLabels=['subset'] + list(overview.columns),\n"
        "    loc='center',\n"
        "    cellLoc='center',\n"
        ")\n"
        "table.auto_set_font_size(False)\n"
        "table.set_fontsize(10)\n"
        "table.scale(1.0, 1.6)\n"
        "ax.set_title('C-MAPSS subset overview', pad=12)\n"
        "fig.tight_layout()\n"
        "fig.savefig(EDA_DIR / '01_dataset_overview.png', bbox_inches='tight', dpi=120)\n"
        "plt.show()"
    ))

    # ----- 2. Engine lifetime distribution -----
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Engine lifetime distribution\n\n"
        "Engines fail at very different cycle counts both within and across subsets. "
        "This variability is why we split by *engine* rather than by *cycle*: shuffling "
        "cycles would let model fit and validation see the same engine and silently leak "
        "information about its degradation trajectory."
    ))
    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=False)\n"
        "for ax, subset in zip(axes.flat, CMAPSS_SUBSETS):\n"
        "    train = pd.read_parquet(INTERIM_DIR / f'{subset}_train.parquet')\n"
        "    lifetimes = train.groupby('unit')['cycle'].max()\n"
        "    ax.hist(lifetimes, bins=25, color='#3b7dd8', edgecolor='white')\n"
        "    ax.set_title(f'{subset}: cycles to failure (n={len(lifetimes)})')\n"
        "    ax.set_xlabel('cycles')\n"
        "    ax.set_ylabel('# engines')\n"
        "    ax.axvline(lifetimes.median(), color='#d62728', linestyle='--', linewidth=1, label=f'median={int(lifetimes.median())}')\n"
        "    ax.legend()\n"
        "fig.suptitle('Engine lifetime distribution per subset', y=1.02)\n"
        "fig.tight_layout()\n"
        "fig.savefig(EDA_DIR / '02_engine_lifetimes.png', bbox_inches='tight', dpi=120)\n"
        "plt.show()"
    ))

    # ----- 3. RUL distribution before/after ceiling -----
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. RUL distribution: raw vs piecewise-linear ceiling\n\n"
        "FD001 training set, every cycle of every engine. The 125-cycle ceiling reflects the "
        "practical reality that engines far from failure are not 'predictably' at any specific "
        "RUL — the regression target should saturate during the healthy phase and only become "
        "informative as degradation appears."
    ))
    cells.append(nbf.v4.new_code_cell(
        "train = load_cmapss_train('FD001')\n"
        "train = add_rul_to_train(train)\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
        "axes[0].hist(train['rul'], bins=50, color='#888888', edgecolor='white')\n"
        "axes[0].set_title('Raw RUL (FD001 train)')\n"
        "axes[0].set_xlabel('RUL (cycles)')\n"
        "axes[0].set_ylabel('count')\n"
        "axes[1].hist(train['rul_clipped'], bins=50, color='#3b7dd8', edgecolor='white')\n"
        "axes[1].axvline(RUL_CEILING, color='#d62728', linestyle='--', linewidth=1, label=f'ceiling={RUL_CEILING}')\n"
        "axes[1].set_title('Clipped RUL (FD001 train)')\n"
        "axes[1].set_xlabel('RUL (cycles)')\n"
        "axes[1].legend()\n"
        "fig.tight_layout()\n"
        "fig.savefig(EDA_DIR / '03_rul_clipping.png', bbox_inches='tight', dpi=120)\n"
        "plt.show()"
    ))

    # ----- 4. Sensor distributions before vs after standardization -----
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Sensor distributions: before vs after standardization\n\n"
        "Box plots of all 21 raw sensors on FD001 train (left) versus the post-normalization "
        "distributions (right). Sensors that look like vertical lines on the raw side are "
        "the flat sensors automatically dropped from the feature set; on the normalized "
        "side only the kept sensors are shown."
    ))
    cells.append(nbf.v4.new_code_cell(
        "raw = load_cmapss_train('FD001')[ALL_SENSOR_COLS]\n"
        "normalized = pd.read_parquet(INTERIM_DIR / 'FD001_train.parquet')\n"
        "kept_cols = [c for c in ALL_SENSOR_COLS if c in normalized.columns and normalized[c].std() > 1e-3]\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n"
        "raw.melt(var_name='sensor', value_name='value').pipe(\n"
        "    lambda d: sns.boxplot(data=d, x='sensor', y='value', ax=axes[0], color='#aaaaaa')\n"
        ")\n"
        "axes[0].set_title('FD001 raw sensors')\n"
        "axes[0].tick_params(axis='x', rotation=90, labelsize=8)\n\n"
        "normalized[kept_cols].melt(var_name='sensor', value_name='value').pipe(\n"
        "    lambda d: sns.boxplot(data=d, x='sensor', y='value', ax=axes[1], color='#3b7dd8')\n"
        ")\n"
        "axes[1].set_title(f'FD001 normalized sensors (kept {len(kept_cols)})')\n"
        "axes[1].tick_params(axis='x', rotation=90, labelsize=8)\n"
        "fig.tight_layout()\n"
        "fig.savefig(EDA_DIR / '04_sensor_distributions.png', bbox_inches='tight', dpi=120)\n"
        "plt.show()"
    ))

    # ----- 5. Sensor degradation trajectories -----
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Sensor degradation trajectories\n\n"
        "Three FD001 training engines (units 1, 50, 100), all kept sensors plotted across "
        "the engine's full lifetime, with a vertical line at failure. Some sensors visibly "
        "drift toward end-of-life; others show no useful trend. The visible-trend sensors are "
        "the ones we'd expect to dominate any RUL model."
    ))
    cells.append(nbf.v4.new_code_cell(
        "normalized = pd.read_parquet(INTERIM_DIR / 'FD001_train.parquet')\n"
        "kept_cols = [c for c in ALL_SENSOR_COLS if c in normalized.columns and normalized[c].std() > 1e-3]\n\n"
        "units_to_plot = [1, 50, 100]\n"
        "fig, axes = plt.subplots(len(units_to_plot), 1, figsize=(12, 3.2 * len(units_to_plot)), sharex=False)\n"
        "for ax, u in zip(axes, units_to_plot):\n"
        "    g = normalized[normalized['unit'] == u].sort_values('cycle')\n"
        "    for col in kept_cols:\n"
        "        ax.plot(g['cycle'], g[col], alpha=0.55, linewidth=0.9, label=col)\n"
        "    failure_cycle = g['cycle'].max()\n"
        "    ax.axvline(failure_cycle, color='#d62728', linestyle='--', linewidth=1)\n"
        "    ax.set_title(f'FD001 unit {u} — lifetime {failure_cycle} cycles')\n"
        "    ax.set_xlabel('cycle')\n"
        "    ax.set_ylabel('normalized value')\n"
        "axes[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=7, ncol=1)\n"
        "fig.tight_layout()\n"
        "fig.savefig(EDA_DIR / '05_degradation_trajectories.png', bbox_inches='tight', dpi=120)\n"
        "plt.show()"
    ))

    # ----- 6. Per-sensor RUL correlation -----
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Per-sensor correlation with clipped RUL (FD001)\n\n"
        "Pearson correlation between each kept FD001 sensor and `rul_clipped`, sorted by "
        "absolute strength. Sensors at the top carry the strongest linear signal about "
        "remaining life and we expect them to be heavily weighted by classical models."
    ))
    cells.append(nbf.v4.new_code_cell(
        "normalized = pd.read_parquet(INTERIM_DIR / 'FD001_train.parquet')\n"
        "kept_cols = [c for c in ALL_SENSOR_COLS if c in normalized.columns and normalized[c].std() > 1e-3]\n\n"
        "corr = normalized[kept_cols + ['rul_clipped']].corr()['rul_clipped'].drop('rul_clipped')\n"
        "corr_sorted = corr.reindex(corr.abs().sort_values(ascending=False).index)\n\n"
        "fig, ax = plt.subplots(figsize=(10, 6))\n"
        "colors = ['#d62728' if v < 0 else '#1b7837' for v in corr_sorted.values]\n"
        "ax.barh(corr_sorted.index[::-1], corr_sorted.values[::-1], color=colors[::-1])\n"
        "ax.set_title('FD001: Pearson correlation between sensor and clipped RUL')\n"
        "ax.set_xlabel('Pearson r')\n"
        "ax.axvline(0, color='black', linewidth=0.7)\n"
        "fig.tight_layout()\n"
        "fig.savefig(EDA_DIR / '06_sensor_rul_correlation.png', bbox_inches='tight', dpi=120)\n"
        "plt.show()"
    ))

    # ----- 7. Operating regime clustering for FD002 -----
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Operating regime clustering (FD002)\n\n"
        "Operational settings 1, 2, 3 plotted pairwise, colored by KMeans regime cluster. "
        "FD002 and FD004 fly under six discrete operating conditions, each of which scales "
        "the sensor responses very differently. Without per-regime normalization, sensor "
        "fluctuations would be dominated by which condition the engine is in rather than "
        "by underlying degradation."
    ))
    cells.append(nbf.v4.new_code_cell(
        "fd002 = pd.read_parquet(INTERIM_DIR / 'FD002_train.parquet')\n"
        "regimes = fd002['regime'].astype(int)\n"
        "n_regimes = regimes.nunique()\n"
        "palette = sns.color_palette('tab10', n_colors=n_regimes)\n\n"
        "pairs = [(0, 1), (0, 2), (1, 2)]\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))\n"
        "for ax, (i, j) in zip(axes, pairs):\n"
        "    xi, xj = OP_SETTING_COLS[i], OP_SETTING_COLS[j]\n"
        "    for r in sorted(regimes.unique()):\n"
        "        mask = regimes == r\n"
        "        ax.scatter(\n"
        "            fd002.loc[mask, xi], fd002.loc[mask, xj],\n"
        "            s=4, alpha=0.5, color=palette[r], label=f'regime {r}'\n"
        "        )\n"
        "    ax.set_xlabel(xi)\n"
        "    ax.set_ylabel(xj)\n"
        "    ax.set_title(f'{xi} vs {xj}')\n"
        "axes[-1].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8)\n"
        "fig.suptitle('FD002: operational settings colored by regime cluster', y=1.02)\n"
        "fig.tight_layout()\n"
        "fig.savefig(EDA_DIR / '07_op_regimes_fd002.png', bbox_inches='tight', dpi=120)\n"
        "plt.show()"
    ))

    # ----- 8. Class balance for binary failure labels -----
    cells.append(nbf.v4.new_markdown_cell(
        "## 8. Class balance for binary failure labels (FD001)\n\n"
        "Per-cycle class balance for `failure_within_{H}` for H ∈ {10, 30, 50}. The shorter the "
        "horizon, the rarer the positive class. Highly imbalanced labels — especially H=10 — make "
        "raw accuracy useless; downstream evaluation will lean on PR-AUC, F1, and per-horizon "
        "calibration."
    ))
    cells.append(nbf.v4.new_code_cell(
        "fd001 = pd.read_parquet(INTERIM_DIR / 'FD001_train.parquet')\n"
        "rows = []\n"
        "for h in PREDICTION_HORIZONS:\n"
        "    pos = (fd001[f'failure_within_{h}'] == 1).sum()\n"
        "    neg = (fd001[f'failure_within_{h}'] == 0).sum()\n"
        "    total = pos + neg\n"
        "    rows.append({'horizon': f'H={h}', 'positive': pos / total, 'negative': neg / total, 'pos_count': int(pos), 'total': int(total)})\n"
        "balance = pd.DataFrame(rows).set_index('horizon')\n"
        "print(balance)\n\n"
        "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
        "balance[['negative', 'positive']].plot(kind='bar', stacked=True, ax=ax, color=['#cccccc', '#d62728'])\n"
        "ax.set_title('FD001 train: class balance for failure_within_H')\n"
        "ax.set_ylabel('proportion of cycles')\n"
        "ax.set_ylim(0, 1)\n"
        "for i, (_, row) in enumerate(balance.iterrows()):\n"
        "    ax.text(i, row['negative'] + row['positive'] / 2,\n"
        "            f\"{row['positive']*100:.1f}%\", ha='center', va='center', color='white', fontsize=9)\n"
        "ax.tick_params(axis='x', rotation=0)\n"
        "fig.tight_layout()\n"
        "fig.savefig(EDA_DIR / '08_class_balance_fd001.png', bbox_inches='tight', dpi=120)\n"
        "plt.show()"
    ))

    nb["cells"] = cells
    out_path = Path("notebooks/01_eda.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        nbf.write(nb, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    build()
