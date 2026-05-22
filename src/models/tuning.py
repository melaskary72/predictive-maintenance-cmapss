"""Optuna hyperparameter tuning for XGBoost models."""
from __future__ import annotations

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score

from src.config import RANDOM_SEED
from src.models.tabular import make_xgb_classifier, make_xgb_regressor

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_xgb_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    seed: int = RANDOM_SEED,
) -> tuple[dict, optuna.Study]:
    """Run Optuna study for XGBoost regressor. Optimizes validation RMSE."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = make_xgb_regressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        rmse = float(np.sqrt(((preds - y_val) ** 2).mean()))
        return rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        study_name="xgb_regressor",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study


def tune_xgb_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    seed: int = RANDOM_SEED,
) -> tuple[dict, optuna.Study]:
    """Run Optuna study for XGBoost classifier. Optimizes validation ROC AUC."""

    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    base_spw = neg_count / max(pos_count, 1.0)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", base_spw * 0.5, base_spw * 2.0
            ),
        }
        model = make_xgb_classifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val, preds))

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        study_name="xgb_classifier",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study
