"""Train LSTM and 1D CNN deep models on FD001 with Optuna tuning.

Trains:
1. LSTM (Optuna tuned, 50 trials)
2. 1D CNN (Optuna tuned, 50 trials)
3. LSTM with auxiliary tabular features (ablation, using best LSTM hyperparams)
4. 1D CNN with auxiliary tabular features (ablation, using best CNN hyperparams)

Saves all artifacts to results/models/phase3/ and writes
results/tables/phase3_results.csv.

Run from project root:
    uv run python scripts/train_deep.py
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    PREDICTION_HORIZONS,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from src.models.deep import DilatedConv1DRegressor, LSTMRegressor
from src.models.deep_training import (
    CMAPSSWindowDataset,
    TrainingConfig,
    predict_deep_model,
    save_checkpoint,
    train_deep_model,
)
from src.models.deep_tuning import tune_cnn, tune_lstm
from src.models.evaluation import (
    classification_metrics,
    regression_metrics,
)
from src.utils.seeds import set_seeds

PRIMARY_HORIZON: int = 30

MODELS_DIR: Path = PROJECT_ROOT / "results" / "models" / "phase3"
TABLES_DIR: Path = PROJECT_ROOT / "results" / "tables"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_fd001_arrays() -> dict:
    """Load FD001 windowed train/val/test plus tabular features for ablation."""
    data = np.load(PROCESSED_DIR / "FD001_windows_train.npz")
    test = np.load(PROCESSED_DIR / "FD001_windows_test.npz")

    X_all = data["X"]
    y_rul_all = data["y_rul"]
    y_clf_all = data[f"y_class_h{PRIMARY_HORIZON}"]
    units = data["unit_ids"]
    train_engines = set(data["train_engine_ids"].tolist())
    val_engines = set(data["val_engine_ids"].tolist())
    train_mask = np.array([u in train_engines for u in units])
    val_mask = np.array([u in val_engines for u in units])

    X_test = test["X"]
    y_rul_test = test["y_rul"]
    y_clf_test = test[f"y_class_h{PRIMARY_HORIZON}"]

    tab_train = pd.read_parquet(PROCESSED_DIR / "FD001_tabular_train.parquet")
    tab_test = pd.read_parquet(PROCESSED_DIR / "FD001_tabular_test.parquet")
    drop_cols = {"unit_id", "last_cycle", "rul_clipped"}
    drop_cols |= {f"failure_within_{h}" for h in PREDICTION_HORIZONS}
    feat_cols = [c for c in tab_train.columns if c not in drop_cols]

    aux_train_full = tab_train[feat_cols].values.astype(np.float32)
    aux_test = tab_test[feat_cols].values.astype(np.float32)

    return {
        "X_train": X_all[train_mask].astype(np.float32),
        "y_rul_train": y_rul_all[train_mask].astype(np.float32),
        "y_clf_train": y_clf_all[train_mask].astype(np.float32),
        "aux_train": aux_train_full[train_mask],
        "X_val": X_all[val_mask].astype(np.float32),
        "y_rul_val": y_rul_all[val_mask].astype(np.float32),
        "y_clf_val": y_clf_all[val_mask].astype(np.float32),
        "aux_val": aux_train_full[val_mask],
        "X_test": X_test.astype(np.float32),
        "y_rul_test": y_rul_test.astype(np.float32),
        "y_clf_test": y_clf_test.astype(np.float32),
        "aux_test": aux_test,
        "n_features": X_all.shape[-1],
        "n_aux_features": aux_train_full.shape[1],
    }


def make_loaders(
    arrays: dict, batch_size: int, use_aux: bool = False
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build DataLoaders for train/val/test."""
    train_aux = arrays["aux_train"] if use_aux else None
    val_aux = arrays["aux_val"] if use_aux else None
    test_aux = arrays["aux_test"] if use_aux else None
    train_ds = CMAPSSWindowDataset(
        arrays["X_train"], arrays["y_rul_train"], arrays["y_clf_train"], train_aux
    )
    val_ds = CMAPSSWindowDataset(
        arrays["X_val"], arrays["y_rul_val"], arrays["y_clf_val"], val_aux
    )
    test_ds = CMAPSSWindowDataset(
        arrays["X_test"], arrays["y_rul_test"], arrays["y_clf_test"], test_aux
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def train_final(
    model: torch.nn.Module,
    arrays: dict,
    best_params: dict,
    pos_weight: float,
    device: torch.device,
    use_aux: bool,
    name: str,
    max_epochs: int = 80,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Train a final model with best Optuna params and longer epoch budget."""
    train_loader, val_loader, test_loader = make_loaders(
        arrays, batch_size=best_params["batch_size"], use_aux=use_aux
    )
    config = TrainingConfig(
        batch_size=best_params["batch_size"],
        epochs=max_epochs,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        clf_loss_weight=best_params["clf_loss_weight"],
        pos_weight=pos_weight,
    )
    print(f"\n=== Training final {name} ===")
    history = train_deep_model(
        model, train_loader, val_loader, config, device, use_aux=use_aux
    )
    model.load_state_dict(history.best_state_dict)
    rul_pred_test, clf_proba_test = predict_deep_model(
        model, test_loader, device, use_aux=use_aux
    )
    save_checkpoint(history, MODELS_DIR / f"{name}.pt")
    return {
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "n_epochs_trained": len(history.epochs),
    }, rul_pred_test, clf_proba_test


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
    print(f"pos_weight (n_neg/n_pos) = {pos_weight:.2f}")

    rows = []

    # ---- LSTM tuning ----
    print("\n" + "=" * 70)
    print("LSTM Optuna tuning (50 trials)")
    print("=" * 70)
    t0 = time.time()
    lstm_params, lstm_study = tune_lstm(
        arrays["X_train"], arrays["y_rul_train"], arrays["y_clf_train"],
        arrays["X_val"], arrays["y_rul_val"], arrays["y_clf_val"],
        device, n_features, pos_weight, n_trials=50, max_epochs=25,
    )
    lstm_tune_seconds = time.time() - t0
    print(f"LSTM best params: {lstm_params}")
    print(f"LSTM tuning took {lstm_tune_seconds:.1f}s")
    with open(MODELS_DIR / "lstm_study.pkl", "wb") as f:
        pickle.dump(lstm_study, f)

    set_seeds(RANDOM_SEED)
    lstm = LSTMRegressor(
        n_features=n_features,
        hidden=lstm_params["hidden"],
        num_layers=lstm_params["num_layers"],
        dropout=lstm_params["dropout"],
    )
    lstm_meta, lstm_rul_test, lstm_clf_test = train_final(
        lstm, arrays, lstm_params, pos_weight, device,
        use_aux=False, name="lstm_final", max_epochs=80,
    )

    rows.append({
        "subset": "FD001", "task": "regression", "model": "lstm",
        **regression_metrics(arrays["y_rul_test"], lstm_rul_test),
        "best_epoch": lstm_meta["best_epoch"],
    })
    rows.append({
        "subset": "FD001", "task": f"classification_h{PRIMARY_HORIZON}",
        "model": "lstm",
        **classification_metrics(arrays["y_clf_test"], lstm_clf_test),
        "best_epoch": lstm_meta["best_epoch"],
    })

    np.save(MODELS_DIR / "lstm_rul_test.npy", lstm_rul_test)
    np.save(MODELS_DIR / "lstm_clf_test.npy", lstm_clf_test)

    # ---- LSTM ablation ----
    print("\n" + "=" * 70)
    print("LSTM ablation (with engineered tabular features)")
    print("=" * 70)
    set_seeds(RANDOM_SEED)
    lstm_aux = LSTMRegressor(
        n_features=n_features,
        hidden=lstm_params["hidden"],
        num_layers=lstm_params["num_layers"],
        dropout=lstm_params["dropout"],
        aux_features=n_aux,
    )
    lstm_aux_meta, lstm_aux_rul_test, lstm_aux_clf_test = train_final(
        lstm_aux, arrays, lstm_params, pos_weight, device,
        use_aux=True, name="lstm_aux_final", max_epochs=80,
    )
    rows.append({
        "subset": "FD001", "task": "regression", "model": "lstm_with_aux",
        **regression_metrics(arrays["y_rul_test"], lstm_aux_rul_test),
        "best_epoch": lstm_aux_meta["best_epoch"],
    })
    rows.append({
        "subset": "FD001", "task": f"classification_h{PRIMARY_HORIZON}",
        "model": "lstm_with_aux",
        **classification_metrics(arrays["y_clf_test"], lstm_aux_clf_test),
        "best_epoch": lstm_aux_meta["best_epoch"],
    })

    # ---- CNN tuning ----
    print("\n" + "=" * 70)
    print("CNN Optuna tuning (50 trials)")
    print("=" * 70)
    t0 = time.time()
    cnn_params, cnn_study = tune_cnn(
        arrays["X_train"], arrays["y_rul_train"], arrays["y_clf_train"],
        arrays["X_val"], arrays["y_rul_val"], arrays["y_clf_val"],
        device, n_features, pos_weight, n_trials=50, max_epochs=25,
    )
    cnn_tune_seconds = time.time() - t0
    print(f"CNN best params: {cnn_params}")
    print(f"CNN tuning took {cnn_tune_seconds:.1f}s")
    with open(MODELS_DIR / "cnn_study.pkl", "wb") as f:
        pickle.dump(cnn_study, f)

    set_seeds(RANDOM_SEED)
    cnn = DilatedConv1DRegressor(
        n_features=n_features,
        n_channels=cnn_params["n_channels"],
        n_blocks=cnn_params["n_blocks"],
        kernel_size=cnn_params["kernel_size"],
        dropout=cnn_params["dropout"],
    )
    cnn_meta, cnn_rul_test, cnn_clf_test = train_final(
        cnn, arrays, cnn_params, pos_weight, device,
        use_aux=False, name="cnn_final", max_epochs=80,
    )
    rows.append({
        "subset": "FD001", "task": "regression", "model": "cnn",
        **regression_metrics(arrays["y_rul_test"], cnn_rul_test),
        "best_epoch": cnn_meta["best_epoch"],
    })
    rows.append({
        "subset": "FD001", "task": f"classification_h{PRIMARY_HORIZON}",
        "model": "cnn",
        **classification_metrics(arrays["y_clf_test"], cnn_clf_test),
        "best_epoch": cnn_meta["best_epoch"],
    })
    np.save(MODELS_DIR / "cnn_rul_test.npy", cnn_rul_test)
    np.save(MODELS_DIR / "cnn_clf_test.npy", cnn_clf_test)

    # ---- CNN ablation ----
    print("\n" + "=" * 70)
    print("CNN ablation (with engineered tabular features)")
    print("=" * 70)
    set_seeds(RANDOM_SEED)
    cnn_aux = DilatedConv1DRegressor(
        n_features=n_features,
        n_channels=cnn_params["n_channels"],
        n_blocks=cnn_params["n_blocks"],
        kernel_size=cnn_params["kernel_size"],
        dropout=cnn_params["dropout"],
        aux_features=n_aux,
    )
    cnn_aux_meta, cnn_aux_rul_test, cnn_aux_clf_test = train_final(
        cnn_aux, arrays, cnn_params, pos_weight, device,
        use_aux=True, name="cnn_aux_final", max_epochs=80,
    )
    rows.append({
        "subset": "FD001", "task": "regression", "model": "cnn_with_aux",
        **regression_metrics(arrays["y_rul_test"], cnn_aux_rul_test),
        "best_epoch": cnn_aux_meta["best_epoch"],
    })
    rows.append({
        "subset": "FD001", "task": f"classification_h{PRIMARY_HORIZON}",
        "model": "cnn_with_aux",
        **classification_metrics(arrays["y_clf_test"], cnn_aux_clf_test),
        "best_epoch": cnn_aux_meta["best_epoch"],
    })

    with open(MODELS_DIR / "best_params.json", "w") as f:
        json.dump({"lstm": lstm_params, "cnn": cnn_params}, f, indent=2)

    np.save(MODELS_DIR / "lstm_aux_rul_test.npy", lstm_aux_rul_test)
    np.save(MODELS_DIR / "lstm_aux_clf_test.npy", lstm_aux_clf_test)
    np.save(MODELS_DIR / "cnn_aux_rul_test.npy", cnn_aux_rul_test)
    np.save(MODELS_DIR / "cnn_aux_clf_test.npy", cnn_aux_clf_test)

    np.save(MODELS_DIR / "y_rul_test.npy", arrays["y_rul_test"])
    np.save(MODELS_DIR / "y_clf_test.npy", arrays["y_clf_test"])

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "phase3_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults table at {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
