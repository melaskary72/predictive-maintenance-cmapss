"""Factory functions for classical tabular models."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor

from src.config import RANDOM_SEED


def make_logreg_classifier(**overrides) -> LogisticRegression:
    """Logistic Regression with class balancing for imbalanced failure labels."""
    defaults = dict(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        solver="lbfgs",
        C=1.0,
    )
    defaults.update(overrides)
    return LogisticRegression(**defaults)


def make_ridge_regressor(**overrides) -> Ridge:
    """Ridge regression for RUL. Uses Ridge instead of plain linear regression
    for better behavior under multicollinearity in engineered features."""
    defaults = dict(alpha=1.0, random_state=RANDOM_SEED)
    defaults.update(overrides)
    return Ridge(**defaults)


def make_rf_classifier(**overrides) -> RandomForestClassifier:
    """Random Forest classifier with class balancing."""
    defaults = dict(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    defaults.update(overrides)
    return RandomForestClassifier(**defaults)


def make_rf_regressor(**overrides) -> RandomForestRegressor:
    """Random Forest regressor for RUL."""
    defaults = dict(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    defaults.update(overrides)
    return RandomForestRegressor(**defaults)


def make_xgb_classifier(**overrides) -> XGBClassifier:
    """XGBoost classifier with sensible defaults; tuning replaces these."""
    defaults = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_SEED,
        tree_method="hist",
        n_jobs=-1,
    )
    defaults.update(overrides)
    return XGBClassifier(**defaults)


def make_xgb_regressor(**overrides) -> XGBRegressor:
    """XGBoost regressor for RUL."""
    defaults = dict(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_SEED,
        tree_method="hist",
        n_jobs=-1,
    )
    defaults.update(overrides)
    return XGBRegressor(**defaults)
