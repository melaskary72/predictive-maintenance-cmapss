"""Evaluate FD001-trained LSTM and CNN on FD002, FD003, FD004 test sets.

This is a generalization test: the deep models were trained only on FD001
(single operating condition, single failure mode). FD002 has 6 operating
conditions, FD003 has two failure modes, FD004 has both. We expect substantial
degradation; the goal is to quantify how much.
"""
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
from src.models.deep import DilatedConv1DRegressor, LSTMRegressor
from src.models.deep_training import (
    CMAPSSWindowDataset,
    predict_deep_model,
)
from src.models.evaluation import classification_metrics, regression_metrics
from src.utils.seeds import set_seeds

PRIMARY_HORIZON = 30
PHASE3_DIR = PROJECT_ROOT / "results" / "models" / "phase3"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    set_seeds(42)
    device = get_device()
    print(f"Device: {device}")

    with open(PHASE3_DIR / "best_params.json") as f:
        best = json.load(f)

    fd001_n_features = 14  # FD001 sensor count after variance filtering

    lstm = LSTMRegressor(
        n_features=fd001_n_features,
        hidden=best["lstm"]["hidden"],
        num_layers=best["lstm"]["num_layers"],
        dropout=best["lstm"]["dropout"],
    ).to(device)
    ckpt = torch.load(PHASE3_DIR / "lstm_final.pt", map_location=device,
                       weights_only=False)
    lstm.load_state_dict(ckpt["state_dict"])

    cnn = DilatedConv1DRegressor(
        n_features=fd001_n_features,
        n_channels=best["cnn"]["n_channels"],
        n_blocks=best["cnn"]["n_blocks"],
        kernel_size=best["cnn"]["kernel_size"],
        dropout=best["cnn"]["dropout"],
    ).to(device)
    ckpt = torch.load(PHASE3_DIR / "cnn_final.pt", map_location=device,
                       weights_only=False)
    cnn.load_state_dict(ckpt["state_dict"])

    fd001_train = np.load(PROCESSED_DIR / "FD001_windows_train.npz")
    fd001_sensors = list(fd001_train["sensor_cols"])
    print(f"FD001 sensor cols ({len(fd001_sensors)}): {fd001_sensors}")

    rows = []
    for subset in ["FD002", "FD003", "FD004"]:
        try:
            test = np.load(PROCESSED_DIR / f"{subset}_windows_test.npz")
            X_test = test["X"].astype(np.float32)
            y_rul = test["y_rul"]
            y_clf = test[f"y_class_h{PRIMARY_HORIZON}"]
            sensor_cols = list(test["sensor_cols"])
            print(f"\n{subset}: X={X_test.shape}, sensors={sensor_cols}")

            if sensor_cols == fd001_sensors:
                X_aligned = X_test
                used_subset = subset
            else:
                # Try to align by intersecting on FD001 sensors
                missing = [s for s in fd001_sensors if s not in sensor_cols]
                if missing:
                    print(
                        f"  Skipping {subset}: missing FD001 sensors {missing} "
                        f"in {subset} feature set."
                    )
                    rows.append({
                        "subset": subset, "task": "regression",
                        "model": "lstm_fd001_transfer", "rmse": float("nan"),
                        "mae": float("nan"), "nasa_score": float("nan"),
                        "n_samples": int(len(y_rul)),
                        "note": f"missing sensors: {missing}",
                    })
                    rows.append({
                        "subset": subset, "task": "regression",
                        "model": "cnn_fd001_transfer", "rmse": float("nan"),
                        "mae": float("nan"), "nasa_score": float("nan"),
                        "n_samples": int(len(y_rul)),
                        "note": f"missing sensors: {missing}",
                    })
                    continue
                idx = [sensor_cols.index(s) for s in fd001_sensors]
                X_aligned = X_test[:, :, idx]
                used_subset = subset
                print(
                    f"  Aligned {subset} to FD001 feature order ({len(idx)} cols)"
                )

            ds = CMAPSSWindowDataset(X_aligned, y_rul.astype(np.float32),
                                     y_clf.astype(np.float32))
            loader = DataLoader(ds, batch_size=512, shuffle=False)

            for name, model in [("lstm_fd001_transfer", lstm),
                                ("cnn_fd001_transfer", cnn)]:
                rul_pred, clf_proba = predict_deep_model(
                    model, loader, device, use_aux=False
                )
                rows.append({
                    "subset": used_subset, "task": "regression", "model": name,
                    **regression_metrics(y_rul, rul_pred),
                })
                rows.append({
                    "subset": used_subset,
                    "task": f"classification_h{PRIMARY_HORIZON}",
                    "model": name,
                    **classification_metrics(y_clf, clf_proba),
                })
        except Exception as exc:
            print(f"  {subset} failed: {exc}")
            rows.append({
                "subset": subset, "task": "regression",
                "model": "lstm_fd001_transfer", "error": str(exc),
            })

    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "phase3_generalization.csv"
    df.to_csv(out_path, index=False)
    print(f"\nGeneralization results at {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
