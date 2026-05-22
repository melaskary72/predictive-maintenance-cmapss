"""Train Transformer encoder on FD001 with Optuna tuning, ablation, and evaluation.

Mirrors scripts/train_deep.py structure but for the Transformer architecture only.
Saves all artifacts to results/models/phase4/ and writes
results/tables/phase4_results.csv.

Run from project root:
    uv run python scripts/train_transformer.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    PREDICTION_HORIZONS,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from src.models.deep import TransformerRegressor
from src.models.deep_training import (
    TrainingConfig,
    predict_deep_model,
    save_checkpoint,
    train_deep_model,
)
from src.models.deep_tuning import tune_transformer
from src.models.evaluation import (
    classification_metrics,
    regression_metrics,
)
from src.utils.seeds import set_seeds

# Reuse the loader machinery from train_deep.py to keep behavior identical
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_deep  # type: ignore  # noqa: E402
from train_deep import (  # type: ignore  # noqa: E402
    get_device,
    load_fd001_arrays,
    make_loaders,
    train_final,
)

PRIMARY_HORIZON: int = 30

MODELS_DIR: Path = PROJECT_ROOT / "results" / "models" / "phase4"
TABLES_DIR: Path = PROJECT_ROOT / "results" / "tables"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# train_final() saves checkpoints to train_deep.MODELS_DIR. Redirect that to
# the Phase 4 directory so artifacts land in the right place.
train_deep.MODELS_DIR = MODELS_DIR


def main() -> None:
    set_seeds(RANDOM_SEED)
    device = get_device()
    print(f"Using device: {device}")

    arrays = load_fd001_arrays()
    n_features = arrays["n_features"]
    n_aux = arrays["n_aux_features"]
    print(f"n_features={n_features}, n_aux={n_aux}")
    print(
        f"Train: {arrays['X_train'].shape}, "
        f"Val: {arrays['X_val'].shape}, Test: {arrays['X_test'].shape}"
    )

    n_pos = float(arrays["y_clf_train"].sum())
    n_neg = float(len(arrays["y_clf_train"]) - n_pos)
    pos_weight = n_neg / max(n_pos, 1.0)
    print(f"pos_weight = {pos_weight:.2f}")

    rows = []

    # ---- Transformer tuning ----
    print("\n" + "=" * 70)
    print("Transformer Optuna tuning (50 trials)")
    print("=" * 70)
    t0 = time.time()
    tx_params, tx_study = tune_transformer(
        arrays["X_train"], arrays["y_rul_train"], arrays["y_clf_train"],
        arrays["X_val"], arrays["y_rul_val"], arrays["y_clf_val"],
        device, n_features, pos_weight, n_trials=50, max_epochs=25,
    )
    print(f"Transformer best params: {tx_params}")
    print(f"Tuning took {time.time()-t0:.1f}s")
    with open(MODELS_DIR / "transformer_study.pkl", "wb") as f:
        pickle.dump(tx_study, f)

    # ---- Final Transformer ----
    set_seeds(RANDOM_SEED)
    tx = TransformerRegressor(
        n_features=n_features,
        d_model=tx_params["d_model"],
        nhead=tx_params["nhead"],
        num_layers=tx_params["num_layers"],
        dim_feedforward=tx_params["dim_feedforward"],
        dropout=tx_params["dropout"],
    )
    tx_meta, tx_rul_test, tx_clf_test = train_final(
        tx, arrays, tx_params, pos_weight, device,
        use_aux=False, name="transformer_final", max_epochs=80,
    )
    rows.append({
        "subset": "FD001", "task": "regression", "model": "transformer",
        **regression_metrics(arrays["y_rul_test"], tx_rul_test),
        "best_epoch": tx_meta["best_epoch"],
    })
    rows.append({
        "subset": "FD001", "task": f"classification_h{PRIMARY_HORIZON}",
        "model": "transformer",
        **classification_metrics(arrays["y_clf_test"], tx_clf_test),
        "best_epoch": tx_meta["best_epoch"],
    })
    np.save(MODELS_DIR / "transformer_rul_test.npy", tx_rul_test)
    np.save(MODELS_DIR / "transformer_clf_test.npy", tx_clf_test)

    # ---- Ablation: Transformer with aux features ----
    print("\n" + "=" * 70)
    print("Transformer ablation (with engineered tabular features)")
    print("=" * 70)
    set_seeds(RANDOM_SEED)
    tx_aux = TransformerRegressor(
        n_features=n_features,
        d_model=tx_params["d_model"],
        nhead=tx_params["nhead"],
        num_layers=tx_params["num_layers"],
        dim_feedforward=tx_params["dim_feedforward"],
        dropout=tx_params["dropout"],
        aux_features=n_aux,
    )
    tx_aux_meta, tx_aux_rul_test, tx_aux_clf_test = train_final(
        tx_aux, arrays, tx_params, pos_weight, device,
        use_aux=True, name="transformer_aux_final", max_epochs=80,
    )
    rows.append({
        "subset": "FD001", "task": "regression", "model": "transformer_with_aux",
        **regression_metrics(arrays["y_rul_test"], tx_aux_rul_test),
        "best_epoch": tx_aux_meta["best_epoch"],
    })
    rows.append({
        "subset": "FD001", "task": f"classification_h{PRIMARY_HORIZON}",
        "model": "transformer_with_aux",
        **classification_metrics(arrays["y_clf_test"], tx_aux_clf_test),
        "best_epoch": tx_aux_meta["best_epoch"],
    })
    np.save(MODELS_DIR / "transformer_aux_rul_test.npy", tx_aux_rul_test)
    np.save(MODELS_DIR / "transformer_aux_clf_test.npy", tx_aux_clf_test)

    # Save best params
    with open(MODELS_DIR / "best_params.json", "w") as f:
        json.dump({"transformer": tx_params}, f, indent=2)

    # Save y_test for downstream scripts (consistent with phase3)
    np.save(MODELS_DIR / "y_rul_test.npy", arrays["y_rul_test"])
    np.save(MODELS_DIR / "y_clf_test.npy", arrays["y_clf_test"])

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "phase4_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults at {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
