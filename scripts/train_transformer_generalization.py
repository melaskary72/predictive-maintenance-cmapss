"""Evaluate FD001-trained Transformer on FD002, FD003, FD004 test sets."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.models.deep import TransformerRegressor
from src.models.deep_training import CMAPSSWindowDataset, predict_deep_model
from src.models.evaluation import classification_metrics, regression_metrics

PRIMARY_HORIZON = 30
MODELS_DIR = PROJECT_ROOT / "results" / "models" / "phase4"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    device = get_device()

    fd001_train = np.load(PROCESSED_DIR / "FD001_windows_train.npz")
    fd001_sensor_cols = list(fd001_train["sensor_cols"])
    n_features = len(fd001_sensor_cols)

    with open(MODELS_DIR / "best_params.json") as f:
        best = json.load(f)["transformer"]

    model = TransformerRegressor(
        n_features=n_features,
        d_model=best["d_model"],
        nhead=best["nhead"],
        num_layers=best["num_layers"],
        dim_feedforward=best["dim_feedforward"],
        dropout=best["dropout"],
    ).to(device)
    ckpt = torch.load(
        MODELS_DIR / "transformer_final.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["state_dict"])

    rows = []
    for subset in ["FD002", "FD003", "FD004"]:
        try:
            data = np.load(PROCESSED_DIR / f"{subset}_windows_test.npz")
            sensor_cols = list(data["sensor_cols"])
            missing = [c for c in fd001_sensor_cols if c not in sensor_cols]
            if missing:
                rows.append({
                    "subset": subset, "task": "regression",
                    "model": "transformer",
                    "note": f"Skipped: {len(missing)} sensors missing",
                })
                rows.append({
                    "subset": subset, "task": "classification",
                    "model": "transformer",
                    "note": f"Skipped: {len(missing)} sensors missing",
                })
                continue

            X = data["X"]
            sensor_idx = [sensor_cols.index(c) for c in fd001_sensor_cols]
            X = X[:, :, sensor_idx].astype(np.float32)

            y_rul = data["y_rul"]
            y_clf = data[f"y_class_h{PRIMARY_HORIZON}"]

            ds = CMAPSSWindowDataset(X, y_rul.astype(np.float32),
                                     y_clf.astype(np.float32))
            loader = DataLoader(ds, batch_size=256, shuffle=False)
            rul_pred, clf_proba = predict_deep_model(
                model, loader, device, use_aux=False
            )
            print(f"{subset} regression RMSE:",
                  np.sqrt(((rul_pred - y_rul) ** 2).mean()))
            rows.append({
                "subset": subset, "task": "regression", "model": "transformer",
                **regression_metrics(y_rul, rul_pred),
            })
            rows.append({
                "subset": subset, "task": "classification",
                "model": "transformer",
                **classification_metrics(y_clf, clf_proba),
            })
        except Exception as exc:
            print(f"Error on {subset}: {exc}")
            rows.append({
                "subset": subset, "task": "error",
                "model": "transformer", "note": str(exc),
            })

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "phase4_generalization.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
