"""Smoke tests for Phase 2 tabular models."""
from __future__ import annotations

import pickle

import pandas as pd
import pytest

from src.config import PROCESSED_DIR, PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "results" / "models" / "phase2"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


@pytest.fixture(scope="module")
def has_phase2_artifacts() -> bool:
    return (MODELS_DIR / "reg_xgboost.pkl").exists()


@pytest.mark.parametrize("name", ["ridge", "random_forest", "xgboost"])
def test_regression_model_loads(name, has_phase2_artifacts):
    if not has_phase2_artifacts:
        pytest.skip("Run scripts/train_tabular.py first")
    path = MODELS_DIR / f"reg_{name}.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)
    assert hasattr(model, "predict")


@pytest.mark.parametrize("name", ["logreg", "random_forest", "xgboost"])
def test_classification_model_loads(name, has_phase2_artifacts):
    if not has_phase2_artifacts:
        pytest.skip("Run scripts/train_tabular.py first")
    path = MODELS_DIR / f"clf_{name}.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)
    assert hasattr(model, "predict_proba")


def test_results_table_exists(has_phase2_artifacts):
    if not has_phase2_artifacts:
        pytest.skip("Run scripts/train_tabular.py first")
    path = TABLES_DIR / "phase2_results.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) >= 6
    assert "model" in df.columns


def test_xgb_predictions_in_valid_range(has_phase2_artifacts):
    if not has_phase2_artifacts:
        pytest.skip("Run scripts/train_tabular.py first")
    test_df = pd.read_parquet(PROCESSED_DIR / "FD001_tabular_test.parquet")
    with open(MODELS_DIR / "feature_cols.pkl", "rb") as f:
        feat_cols = pickle.load(f)
    with open(MODELS_DIR / "clf_xgboost.pkl", "rb") as f:
        clf = pickle.load(f)
    proba = clf.predict_proba(test_df[feat_cols])[:, 1]
    assert proba.min() >= 0
    assert proba.max() <= 1
