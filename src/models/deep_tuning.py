"""Optuna hyperparameter tuning for deep models."""
from __future__ import annotations

import optuna
import torch
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader

from src.config import RANDOM_SEED
from src.models.deep import (
    DilatedConv1DRegressor,
    LSTMRegressor,
    TransformerRegressor,
)
from src.models.deep_training import (
    CMAPSSWindowDataset,
    TrainingConfig,
    train_deep_model,
)
from src.utils.seeds import set_seeds

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_loaders(X_tr, y_rul_tr, y_clf_tr, X_va, y_rul_va, y_clf_va, batch_size):
    train_ds = CMAPSSWindowDataset(X_tr, y_rul_tr, y_clf_tr)
    val_ds = CMAPSSWindowDataset(X_va, y_rul_va, y_clf_va)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, drop_last=False)
    return train_loader, val_loader


def tune_lstm(
    X_tr, y_rul_tr, y_clf_tr, X_va, y_rul_va, y_clf_va,
    device: torch.device, n_features: int, pos_weight: float,
    n_trials: int = 50, max_epochs: int = 30,
) -> tuple[dict, optuna.Study]:
    """Tune LSTM. Optimizes validation total loss (reg + lambda*clf)."""

    def objective(trial: optuna.Trial) -> float:
        set_seeds(RANDOM_SEED)
        params = {
            "hidden": trial.suggest_categorical("hidden", [32, 64, 96, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "clf_loss_weight": trial.suggest_float(
                "clf_loss_weight", 0.1, 5.0, log=True
            ),
        }
        model = LSTMRegressor(
            n_features=n_features,
            hidden=params["hidden"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
        )
        config = TrainingConfig(
            batch_size=params["batch_size"],
            epochs=max_epochs,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            clf_loss_weight=params["clf_loss_weight"],
            pos_weight=pos_weight,
        )
        train_loader, val_loader = _make_loaders(
            X_tr, y_rul_tr, y_clf_tr, X_va, y_rul_va, y_clf_va,
            params["batch_size"],
        )
        history = train_deep_model(
            model, train_loader, val_loader, config, device,
            use_aux=False, verbose=False,
        )
        return history.best_val_loss

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RANDOM_SEED),
        study_name="lstm",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study


def tune_cnn(
    X_tr, y_rul_tr, y_clf_tr, X_va, y_rul_va, y_clf_va,
    device: torch.device, n_features: int, pos_weight: float,
    n_trials: int = 50, max_epochs: int = 30,
) -> tuple[dict, optuna.Study]:
    """Tune dilated 1D CNN."""

    def objective(trial: optuna.Trial) -> float:
        set_seeds(RANDOM_SEED)
        params = {
            "n_channels": trial.suggest_categorical("n_channels", [32, 64, 96, 128]),
            "n_blocks": trial.suggest_int("n_blocks", 2, 5),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "clf_loss_weight": trial.suggest_float(
                "clf_loss_weight", 0.1, 5.0, log=True
            ),
        }
        model = DilatedConv1DRegressor(
            n_features=n_features,
            n_channels=params["n_channels"],
            n_blocks=params["n_blocks"],
            kernel_size=params["kernel_size"],
            dropout=params["dropout"],
        )
        config = TrainingConfig(
            batch_size=params["batch_size"],
            epochs=max_epochs,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            clf_loss_weight=params["clf_loss_weight"],
            pos_weight=pos_weight,
        )
        train_loader, val_loader = _make_loaders(
            X_tr, y_rul_tr, y_clf_tr, X_va, y_rul_va, y_clf_va,
            params["batch_size"],
        )
        history = train_deep_model(
            model, train_loader, val_loader, config, device,
            use_aux=False, verbose=False,
        )
        return history.best_val_loss

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RANDOM_SEED),
        study_name="cnn",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study


def tune_transformer(
    X_tr, y_rul_tr, y_clf_tr, X_va, y_rul_va, y_clf_va,
    device: torch.device, n_features: int, pos_weight: float,
    n_trials: int = 50, max_epochs: int = 25,
) -> tuple[dict, optuna.Study]:
    """Tune Transformer encoder. Optimizes validation total loss.

    Optuna does not allow dynamic categorical choices that depend on other
    parameters, so we give each d_model its own conditional nhead parameter
    keyed as nhead_d{d_model} and normalize at the end.
    """

    def objective(trial: optuna.Trial) -> float:
        set_seeds(RANDOM_SEED)
        d_model = trial.suggest_categorical("d_model", [32, 64, 96, 128])
        nhead_choices = [h for h in [2, 4, 8] if d_model % h == 0]
        nhead = trial.suggest_categorical(f"nhead_d{d_model}", nhead_choices)
        params = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "dim_feedforward": trial.suggest_categorical(
                "dim_feedforward", [64, 128, 256]
            ),
            "dropout": trial.suggest_float("dropout", 0.05, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float(
                "weight_decay", 1e-6, 1e-3, log=True
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", [128, 256, 512]
            ),
            "clf_loss_weight": trial.suggest_float(
                "clf_loss_weight", 0.1, 5.0, log=True
            ),
        }
        model = TransformerRegressor(
            n_features=n_features,
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_layers=params["num_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params["dropout"],
        )
        config = TrainingConfig(
            batch_size=params["batch_size"],
            epochs=max_epochs,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            clf_loss_weight=params["clf_loss_weight"],
            pos_weight=pos_weight,
        )
        train_loader, val_loader = _make_loaders(
            X_tr, y_rul_tr, y_clf_tr, X_va, y_rul_va, y_clf_va,
            params["batch_size"],
        )
        history = train_deep_model(
            model, train_loader, val_loader, config, device,
            use_aux=False, verbose=False,
        )
        trial.set_user_attr("resolved_nhead", nhead)
        return history.best_val_loss

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=RANDOM_SEED),
        study_name="transformer",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    nhead_key = f"nhead_d{best['d_model']}"
    if nhead_key in best:
        best["nhead"] = best.pop(nhead_key)
    return best, study
