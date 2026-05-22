"""Training loop, dataset utilities, and checkpoint management for deep models."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import RANDOM_SEED


@dataclass
class TrainingConfig:
    """Hyperparameters for a single training run."""
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 8
    lr_factor: float = 0.5
    lr_patience: int = 4
    clf_loss_weight: float = 1.0
    pos_weight: float = 1.0
    seed: int = RANDOM_SEED


class CMAPSSWindowDataset(Dataset):
    """Dataset over windowed C-MAPSS data with optional auxiliary tabular features."""

    def __init__(
        self,
        X: np.ndarray,
        y_rul: np.ndarray,
        y_clf: np.ndarray,
        aux: Optional[np.ndarray] = None,
    ):
        self.X = torch.from_numpy(X).float()
        self.y_rul = torch.from_numpy(y_rul).float()
        self.y_clf = torch.from_numpy(y_clf).float()
        self.aux = torch.from_numpy(aux).float() if aux is not None else None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        if self.aux is not None:
            return self.X[idx], self.aux[idx], self.y_rul[idx], self.y_clf[idx]
        return self.X[idx], self.y_rul[idx], self.y_clf[idx]


@dataclass
class EpochMetrics:
    train_loss: float
    val_loss: float
    val_reg_loss: float
    val_clf_loss: float
    val_rmse: float
    val_roc_auc: float
    epoch_seconds: float


@dataclass
class TrainingHistory:
    epochs: list[EpochMetrics] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    best_state_dict: Optional[dict] = None


def _move_batch(batch, device):
    """Move a batch (tuple of tensors) to device."""
    return tuple(t.to(device) for t in batch)


def train_deep_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    use_aux: bool = False,
    verbose: bool = True,
) -> TrainingHistory:
    """Train a dual-head model with early stopping, lr scheduling, gradient clipping.

    Returns a TrainingHistory containing per-epoch metrics and the best-checkpoint
    state dict.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience,
    )
    reg_loss_fn = nn.SmoothL1Loss()
    clf_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config.pos_weight], device=device)
    )

    history = TrainingHistory()
    bad_epochs = 0

    from sklearn.metrics import roc_auc_score

    for epoch in range(config.epochs):
        t0 = time.time()
        model.train()
        running_train = 0.0
        n_train = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            if use_aux:
                X, aux, y_rul, y_clf = batch
                reg_pred, clf_pred = model(X, aux)
            else:
                X, y_rul, y_clf = batch
                reg_pred, clf_pred = model(X)
            loss_reg = reg_loss_fn(reg_pred, y_rul)
            loss_clf = clf_loss_fn(clf_pred, y_clf)
            loss = loss_reg + config.clf_loss_weight * loss_clf
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            running_train += loss.item() * X.size(0)
            n_train += X.size(0)

        model.eval()
        val_reg, val_clf, n_val = 0.0, 0.0, 0
        all_reg_pred, all_y_rul = [], []
        all_clf_pred, all_y_clf = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                if use_aux:
                    X, aux, y_rul, y_clf = batch
                    reg_pred, clf_pred = model(X, aux)
                else:
                    X, y_rul, y_clf = batch
                    reg_pred, clf_pred = model(X)
                loss_reg = reg_loss_fn(reg_pred, y_rul)
                loss_clf = clf_loss_fn(clf_pred, y_clf)
                val_reg += loss_reg.item() * X.size(0)
                val_clf += loss_clf.item() * X.size(0)
                n_val += X.size(0)
                all_reg_pred.append(reg_pred.cpu().numpy())
                all_y_rul.append(y_rul.cpu().numpy())
                all_clf_pred.append(torch.sigmoid(clf_pred).cpu().numpy())
                all_y_clf.append(y_clf.cpu().numpy())

        val_reg /= n_val
        val_clf /= n_val
        val_loss = val_reg + config.clf_loss_weight * val_clf
        all_reg_pred = np.concatenate(all_reg_pred)
        all_y_rul = np.concatenate(all_y_rul)
        all_clf_pred = np.concatenate(all_clf_pred)
        all_y_clf = np.concatenate(all_y_clf)
        val_rmse = float(np.sqrt(((all_reg_pred - all_y_rul) ** 2).mean()))
        try:
            val_auc = float(roc_auc_score(all_y_clf, all_clf_pred))
        except ValueError:
            val_auc = float("nan")

        scheduler.step(val_loss)
        epoch_metrics = EpochMetrics(
            train_loss=running_train / n_train,
            val_loss=val_loss,
            val_reg_loss=val_reg,
            val_clf_loss=val_clf,
            val_rmse=val_rmse,
            val_roc_auc=val_auc,
            epoch_seconds=time.time() - t0,
        )
        history.epochs.append(epoch_metrics)

        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            history.best_state_dict = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose:
            print(
                f"Epoch {epoch:3d} | train {epoch_metrics.train_loss:.4f} | "
                f"val {val_loss:.4f} | rmse {val_rmse:.3f} | "
                f"auc {val_auc:.3f} | {epoch_metrics.epoch_seconds:.1f}s"
            )

        if bad_epochs >= config.patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    return history


def predict_deep_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_aux: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a loader. Returns (rul_preds, clf_probs)."""
    model.eval()
    all_reg, all_clf = [], []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            if use_aux:
                X, aux, *_ = batch
                reg_pred, clf_pred = model(X, aux)
            else:
                X, *_ = batch
                reg_pred, clf_pred = model(X)
            all_reg.append(reg_pred.cpu().numpy())
            all_clf.append(torch.sigmoid(clf_pred).cpu().numpy())
    return np.concatenate(all_reg), np.concatenate(all_clf)


def save_checkpoint(history: TrainingHistory, path: Path) -> None:
    """Save best state dict + minimal metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": history.best_state_dict,
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "n_epochs_trained": len(history.epochs),
        "history": [vars(e) for e in history.epochs],
    }
    torch.save(payload, path)
