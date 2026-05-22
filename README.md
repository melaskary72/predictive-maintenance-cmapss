# Predictive Maintenance on NASA C-MAPSS Turbofan Engines

CPSC 393 Machine Learning Final Project, Spring 2026
Chapman University
Author: Mohamed El Askary

## Project Overview

A comparative study of classical and deep learning models for predicting
Remaining Useful Life (RUL) and imminent failure on the NASA C-MAPSS
Turbofan Engine Degradation dataset. Models compared: Logistic Regression,
Random Forest, XGBoost, LSTM, 1D CNN, and Transformer encoder, evaluated
across three prediction horizons (10, 30, 50 cycles).

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/melaskary72/predictive-maintenance-cmapss.git
cd predictive-maintenance-cmapss
uv sync
```

## Data

The raw C-MAPSS dataset is the [NASA Turbofan Jet Engine Data Set](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) on Kaggle (CC0 Public Domain).

To set up the dataset:
1. Download `archive.zip` from the Kaggle link above.
2. Place it in `~/Downloads/`.
3. Run `unzip -j ~/Downloads/archive.zip 'CMaps/*' -d data/raw/cmapss/` from the project root.
4. Run the preparation pipeline: `uv run python scripts/prepare_data.py`.

This produces `data/processed/{subset}_windows_train.npz`, `_windows_test.npz`, and `_tabular_train/test.parquet` files for each of the four subsets, plus a `manifest.json` summarizing the pipeline output.

## Repository Structure

```
src/
├── config.py                # Central configuration
├── data/                    # Data loading, preprocessing, windows, splits
├── models/
│   ├── tabular.py           # Phase 2 classical model factories
│   ├── tuning.py            # Phase 2 Optuna search
│   ├── evaluation.py        # Shared metrics + diagnostic plots
│   ├── deep.py              # PyTorch model definitions (LSTM, CNN, Transformer)
│   ├── deep_training.py     # Training loop, dataset, checkpoints
│   ├── deep_tuning.py       # Optuna search for LSTM, CNN, Transformer
│   └── ensemble.py          # Ensemble averaging utilities
└── utils/                   # Seeds, plotting helpers
scripts/
├── prepare_data.py          # Data preparation
├── train_tabular.py         # Phase 2 training
├── evaluate_tabular.py      # Phase 2 evaluation
├── train_deep.py            # Phase 3 training (Optuna + final + ablations)
├── run_ensemble.py          # Phase 3 ensemble
├── evaluate_deep.py         # Phase 3 diagnostics, calibration, SHAP
├── evaluate_generalization.py  # Phase 3 cross-subset transfer
├── train_transformer.py     # Phase 4 training (Optuna + final + ablation)
├── train_transformer_generalization.py  # Phase 4 cross-subset transfer
├── evaluate_transformer.py  # Phase 4 diagnostics, calibration, SHAP
└── run_full_ensemble.py     # Phase 4 4-way ensemble + master comparison
notebooks/
└── 01_eda.ipynb             # Exploratory data analysis
tests/
├── test_data_pipeline.py    # Phase 1 sanity tests
├── test_phase2.py           # Phase 2 smoke tests
├── test_phase3.py           # Phase 3 smoke tests
└── test_phase4.py           # Phase 4 smoke tests
```

## Tests

```bash
uv run pytest tests/ -v
```

## Project Phases

This is a multi-phase project:
- **Phase 1:** data pipeline, EDA, sanity tests
- **Phase 2:** classical baselines (Logistic Regression, Random Forest, XGBoost)
- **Phase 3:** deep learning models (LSTM, 1D CNN, ensemble, SHAP, calibration)
- **Phase 4 (current):** Transformer encoder, full master ensemble comparison
- **Phase 5:** horizon sweep, AI4I generalization, SHAP, calibration analysis
- **Phase 6:** final report, slide deck, polished README

## Phase 2: Tabular Baselines

Trained Logistic Regression, Random Forest, and XGBoost on FD001 engineered window features. XGBoost hyperparameters tuned via Optuna with 50 trials per task (TPE sampler) on a held-out engine-level validation set.

To reproduce:

```bash
uv run python scripts/train_tabular.py    # ~15-25 min
uv run python scripts/evaluate_tabular.py # ~1 min
```

Results saved to `results/tables/phase2_results.csv` and figures to `results/figures/phase2/`. The FD001-tuned XGBoost models are also evaluated on FD002, FD003, FD004 test sets to measure how well a single-condition model generalizes.

## Phase 3: Deep Learning Models

PyTorch deep models with a dual-head architecture (shared backbone, separate regression and classification heads):

- Stacked LSTM, hyperparameters tuned via Optuna (50 trials)
- Dilated 1D CNN (TCN-style), hyperparameters tuned via Optuna (50 trials)
- Both architectures also trained with auxiliary engineered tabular features as an ablation, to test whether deep models genuinely capture temporal structure beyond what hand-engineered features express
- Ensemble of XGBoost + LSTM + CNN via average of predictions
- SHAP analysis of LSTM feature importance via `GradientExplainer` (more compatible with LSTMs than `DeepExplainer` on the MPS backend)
- Calibration analysis (reliability diagrams + Brier score) for all classifiers
- Generalization evaluation: FD001-trained best LSTM and CNN evaluated on FD002, FD003, FD004 (where sensor sets allow)

To reproduce:

```bash
uv run python scripts/train_deep.py             # ~80-110 min on Apple Silicon MPS
uv run python scripts/run_ensemble.py           # <1 min
uv run python scripts/evaluate_deep.py          # ~3 min
uv run python scripts/evaluate_generalization.py  # <1 min
```

Models saved to `results/models/phase3/`. Results to `results/tables/phase3_results.csv`, `results/tables/phase3_ensemble_results.csv`, and `results/tables/phase3_generalization.csv`. Figures to `results/figures/phase3/`.

## Phase 4: Transformer Encoder

Transformer encoder with learned positional embeddings, GELU activation, and pre-norm residual connections, completing the architecture taxonomy: classical (linear, tree), recurrent (LSTM), convolutional (TCN-style dilated CNN), and attention-based (Transformer).

Hyperparameters tuned via Optuna (50 trials), with constrained sampling to ensure `d_model` is always divisible by `nhead`. Same dual-head output (RUL regression + failure classification) and same auxiliary-feature ablation as Phase 3. The Transformer is also added to the ensemble: Phase 4 produces both per-architecture metrics and a master comparison table consolidating XGBoost, LSTM, CNN, Transformer, and every ensemble combination.

To reproduce:

```bash
uv run python scripts/train_transformer.py                 # ~50-65 min
uv run python scripts/train_transformer_generalization.py  # <1 min
uv run python scripts/run_full_ensemble.py                 # <1 min
uv run python scripts/evaluate_transformer.py              # ~2 min
```

Models saved to `results/models/phase4/`. Results to `results/tables/phase4_results.csv`, `results/tables/phase4_generalization.csv`, `results/tables/phase4_ensemble_results.csv`, and the consolidated `results/tables/master_comparison.csv`. Figures to `results/figures/phase4/`.

## Phase 5: Horizon Sweep and Cross-Dataset Generalization

**Horizon sweep:** classification metrics for all model families across three failure-prediction horizons (10, 30, 50 cycles). XGBoost is retrained at each horizon with light Optuna tuning (25 trials per horizon) since optimal hyperparameters differ with class imbalance. Deep models (LSTM, CNN, Transformer) reuse predictions from their h=30 training and are evaluated against alternate-horizon labels to test representation transfer across horizons.

**AI4I 2020 cross-dataset generalization:** an XGBoost classifier trained on the AI4I 2020 Predictive Maintenance Dataset (UCI) with the same Optuna methodology, including a 5-fold cross-validated robustness check. Tests whether the engineering and methodology transfer to a structurally different industrial PM problem.

To reproduce:

```bash
uv run python scripts/horizon_sweep.py    # ~10 min
uv run python scripts/train_ai4i.py       # ~7 min
```

Outputs: `results/tables/horizon_sweep.csv`, `results/tables/ai4i_results.csv`, plus 5+ figures in `results/figures/phase5/`.

## License

Code: MIT. Dataset: CC0 (NASA Prognostics Center of Excellence via Kaggle mirror).
