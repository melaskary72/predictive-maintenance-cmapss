"""Train Logistic Regression, Random Forest, and XGBoost on FD001 tabular features.

Trains both regression (RUL) and classification (failure within 30 cycles) tasks.
Uses Optuna with 50 trials for XGBoost. Light tuning for RF and LR.

Saves all artifacts to results/models/phase2/ and results to
results/tables/phase2_results.csv.

Run from project root:
    uv run python scripts/train_tabular.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    PREDICTION_HORIZONS,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from src.models.evaluation import (
    classification_metrics,
    regression_metrics,
)
from src.models.tabular import (
    make_logreg_classifier,
    make_rf_classifier,
    make_rf_regressor,
    make_ridge_regressor,
    make_xgb_classifier,
    make_xgb_regressor,
)
from src.models.tuning import tune_xgb_classifier, tune_xgb_regressor
from src.utils.seeds import set_seeds

PRIMARY_HORIZON: int = 30

MODELS_DIR: Path = PROJECT_ROOT / "results" / "models" / "phase2"
TABLES_DIR: Path = PROJECT_ROOT / "results" / "tables"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_subset_arrays(subset: str) -> dict:
    """Load processed tabular features and engine splits for a subset."""
    train_df = pd.read_parquet(
        PROCESSED_DIR / f"{subset}_tabular_train.parquet"
    )
    test_df = pd.read_parquet(
        PROCESSED_DIR / f"{subset}_tabular_test.parquet"
    )

    windows = np.load(PROCESSED_DIR / f"{subset}_windows_train.npz")
    train_engines = set(windows["train_engine_ids"].tolist())
    val_engines = set(windows["val_engine_ids"].tolist())

    train_mask = train_df["unit_id"].isin(train_engines).values
    val_mask = train_df["unit_id"].isin(val_engines).values

    drop_cols = {"unit_id", "last_cycle", "rul_clipped"}
    drop_cols |= {f"failure_within_{h}" for h in PREDICTION_HORIZONS}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    return {
        "train_df": train_df,
        "test_df": test_df,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "feature_cols": feature_cols,
    }


def split_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract X, y from a DataFrame with optional row mask."""
    if mask is not None:
        df = df[mask]
    return df[feature_cols].values, df[target].values


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def train_regression_models(
    X_train, y_train, X_val, y_val, X_test, y_test, n_trials: int = 50
) -> dict:
    """Train and evaluate all three regressors. Returns metrics + fitted models."""
    results = {}

    print("\n--- Ridge regression ---")
    t0 = time.time()
    ridge = make_ridge_regressor(alpha=1.0)
    ridge.fit(X_train, y_train)
    results["ridge"] = {
        "model": ridge,
        "val_metrics": regression_metrics(y_val, ridge.predict(X_val)),
        "test_metrics": regression_metrics(y_test, ridge.predict(X_test)),
        "train_seconds": time.time() - t0,
    }
    print(f"  Val RMSE: {results['ridge']['val_metrics']['rmse']:.3f}")
    print(f"  Test RMSE: {results['ridge']['test_metrics']['rmse']:.3f}")

    print("\n--- Random Forest regression ---")
    t0 = time.time()
    rf = make_rf_regressor(n_estimators=400)
    rf.fit(X_train, y_train)
    results["random_forest"] = {
        "model": rf,
        "val_metrics": regression_metrics(y_val, rf.predict(X_val)),
        "test_metrics": regression_metrics(y_test, rf.predict(X_test)),
        "train_seconds": time.time() - t0,
    }
    print(f"  Val RMSE: {results['random_forest']['val_metrics']['rmse']:.3f}")
    print(f"  Test RMSE: {results['random_forest']['test_metrics']['rmse']:.3f}")

    print(f"\n--- XGBoost regression (Optuna, {n_trials} trials) ---")
    t0 = time.time()
    best_params, study = tune_xgb_regressor(
        X_train, y_train, X_val, y_val, n_trials=n_trials
    )
    print(f"  Best params: {best_params}")
    print(f"  Best val RMSE: {study.best_value:.3f}")

    xgb = make_xgb_regressor(**best_params)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    results["xgboost"] = {
        "model": xgb,
        "val_metrics": regression_metrics(y_val, xgb.predict(X_val)),
        "test_metrics": regression_metrics(y_test, xgb.predict(X_test)),
        "best_params": best_params,
        "train_seconds": time.time() - t0,
        "study": study,
    }
    print(f"  Test RMSE: {results['xgboost']['test_metrics']['rmse']:.3f}")
    print(f"  Test NASA score: {results['xgboost']['test_metrics']['nasa_score']:.1f}")

    return results


def train_classification_models(
    X_train, y_train, X_val, y_val, X_test, y_test, n_trials: int = 50
) -> dict:
    """Train and evaluate all three classifiers. Returns metrics + fitted models."""
    results = {}

    print("\n--- Logistic Regression classification ---")
    t0 = time.time()
    lr = make_logreg_classifier(C=1.0)
    lr.fit(X_train, y_train)
    val_proba = lr.predict_proba(X_val)[:, 1]
    test_proba = lr.predict_proba(X_test)[:, 1]
    results["logreg"] = {
        "model": lr,
        "val_metrics": classification_metrics(y_val, val_proba),
        "test_metrics": classification_metrics(y_test, test_proba),
        "test_proba": test_proba,
        "train_seconds": time.time() - t0,
    }
    print(f"  Val ROC AUC: {results['logreg']['val_metrics']['roc_auc']:.3f}")
    print(f"  Test ROC AUC: {results['logreg']['test_metrics']['roc_auc']:.3f}")

    print("\n--- Random Forest classification ---")
    t0 = time.time()
    rf = make_rf_classifier(n_estimators=400)
    rf.fit(X_train, y_train)
    val_proba = rf.predict_proba(X_val)[:, 1]
    test_proba = rf.predict_proba(X_test)[:, 1]
    results["random_forest"] = {
        "model": rf,
        "val_metrics": classification_metrics(y_val, val_proba),
        "test_metrics": classification_metrics(y_test, test_proba),
        "test_proba": test_proba,
        "train_seconds": time.time() - t0,
    }
    print(f"  Val ROC AUC: {results['random_forest']['val_metrics']['roc_auc']:.3f}")
    print(f"  Test ROC AUC: {results['random_forest']['test_metrics']['roc_auc']:.3f}")

    print(f"\n--- XGBoost classification (Optuna, {n_trials} trials) ---")
    t0 = time.time()
    best_params, study = tune_xgb_classifier(
        X_train, y_train, X_val, y_val, n_trials=n_trials
    )
    print(f"  Best params: {best_params}")
    print(f"  Best val ROC AUC: {study.best_value:.3f}")

    xgb = make_xgb_classifier(**best_params)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_proba = xgb.predict_proba(X_val)[:, 1]
    test_proba = xgb.predict_proba(X_test)[:, 1]
    results["xgboost"] = {
        "model": xgb,
        "val_metrics": classification_metrics(y_val, val_proba),
        "test_metrics": classification_metrics(y_test, test_proba),
        "test_proba": test_proba,
        "best_params": best_params,
        "train_seconds": time.time() - t0,
        "study": study,
    }
    print(f"  Test ROC AUC: {results['xgboost']['test_metrics']['roc_auc']:.3f}")
    print(f"  Test PR AUC: {results['xgboost']['test_metrics']['pr_auc']:.3f}")

    return results


def evaluate_xgb_on_other_subsets(
    xgb_reg, xgb_clf, primary_horizon: int = PRIMARY_HORIZON
) -> list[dict]:
    """Apply FD001-tuned XGBoost to FD002, FD003, FD004 test sets without retuning."""
    rows = []
    for subset in ["FD002", "FD003", "FD004"]:
        print(f"\n--- Generalization eval on {subset} ---")
        try:
            arrays = load_subset_arrays(subset)
            test_df = arrays["test_df"]
            feat_cols = arrays["feature_cols"]
            xgb_reg_features = list(xgb_reg.feature_names_in_) if hasattr(
                xgb_reg, "feature_names_in_"
            ) else feat_cols
            common = [c for c in xgb_reg_features if c in feat_cols]
            missing = set(xgb_reg_features) - set(feat_cols)
            if missing:
                print(
                    f"  WARNING: {len(missing)} features missing in {subset}, "
                    f"skipping generalization for this subset. "
                    f"(Missing: {sorted(missing)[:5]}...)"
                )
                rows.append({
                    "subset": subset,
                    "task": "regression",
                    "note": f"Skipped: {len(missing)} feature mismatch",
                })
                rows.append({
                    "subset": subset,
                    "task": "classification",
                    "note": f"Skipped: {len(missing)} feature mismatch",
                })
                continue

            X_test = test_df[common].values
            y_test_rul = test_df["rul_clipped"].values
            y_test_clf = test_df[f"failure_within_{primary_horizon}"].values

            reg_metrics = regression_metrics(y_test_rul, xgb_reg.predict(X_test))
            clf_metrics = classification_metrics(
                y_test_clf, xgb_clf.predict_proba(X_test)[:, 1]
            )
            print(f"  Regression test RMSE: {reg_metrics['rmse']:.3f}")
            print(f"  Classification test ROC AUC: {clf_metrics['roc_auc']:.3f}")

            rows.append({
                "subset": subset, "task": "regression", **reg_metrics
            })
            rows.append({
                "subset": subset, "task": "classification", **clf_metrics
            })
        except Exception as exc:
            print(f"  ERROR on {subset}: {exc}")
            rows.append({"subset": subset, "task": "error", "note": str(exc)})
    return rows


def main() -> None:
    set_seeds(RANDOM_SEED)
    print("=" * 70)
    print("Phase 2: Tabular Baselines on FD001")
    print("=" * 70)

    arrays = load_subset_arrays("FD001")
    train_df = arrays["train_df"]
    test_df = arrays["test_df"]
    train_mask = arrays["train_mask"]
    val_mask = arrays["val_mask"]
    feat_cols = arrays["feature_cols"]
    print(f"\nFeature columns: {len(feat_cols)}")
    print(f"Train rows: {int(train_mask.sum())}")
    print(f"Val rows:   {int(val_mask.sum())}")
    print(f"Test rows:  {len(test_df)}")

    print("\n" + "=" * 70)
    print("REGRESSION TASK: Predict clipped RUL")
    print("=" * 70)
    X_tr, y_tr = split_xy(train_df, feat_cols, "rul_clipped", train_mask)
    X_va, y_va = split_xy(train_df, feat_cols, "rul_clipped", val_mask)
    X_te, y_te = split_xy(test_df, feat_cols, "rul_clipped")

    X_tr_df = train_df[train_mask][feat_cols]
    X_va_df = train_df[val_mask][feat_cols]
    X_te_df = test_df[feat_cols]

    reg_results = train_regression_models(
        X_tr_df.values, y_tr, X_va_df.values, y_va, X_te_df.values, y_te
    )

    print("\n" + "=" * 70)
    print(f"CLASSIFICATION TASK: failure_within_{PRIMARY_HORIZON}")
    print("=" * 70)
    target_clf = f"failure_within_{PRIMARY_HORIZON}"
    _, y_tr_c = split_xy(train_df, feat_cols, target_clf, train_mask)
    _, y_va_c = split_xy(train_df, feat_cols, target_clf, val_mask)
    _, y_te_c = split_xy(test_df, feat_cols, target_clf)

    print(f"  Train pos rate: {y_tr_c.mean():.3f}")
    print(f"  Val pos rate:   {y_va_c.mean():.3f}")
    print(f"  Test pos rate:  {y_te_c.mean():.3f}")

    clf_results = train_classification_models(
        X_tr_df.values, y_tr_c, X_va_df.values, y_va_c, X_te_df.values, y_te_c
    )

    # Refit XGBoost models with feature names so we can identify columns later
    xgb_reg_named = make_xgb_regressor(**reg_results["xgboost"]["best_params"])
    xgb_reg_named.fit(X_tr_df, y_tr, eval_set=[(X_va_df, y_va)], verbose=False)
    reg_results["xgboost"]["model"] = xgb_reg_named

    xgb_clf_named = make_xgb_classifier(**clf_results["xgboost"]["best_params"])
    xgb_clf_named.fit(X_tr_df, y_tr_c, eval_set=[(X_va_df, y_va_c)], verbose=False)
    clf_results["xgboost"]["model"] = xgb_clf_named

    print("\n--- Saving artifacts ---")
    for name, info in reg_results.items():
        save_pickle(info["model"], MODELS_DIR / f"reg_{name}.pkl")
    for name, info in clf_results.items():
        save_pickle(info["model"], MODELS_DIR / f"clf_{name}.pkl")
    save_pickle(reg_results["xgboost"]["study"], MODELS_DIR / "xgb_reg_study.pkl")
    save_pickle(clf_results["xgboost"]["study"], MODELS_DIR / "xgb_clf_study.pkl")
    save_pickle(feat_cols, MODELS_DIR / "feature_cols.pkl")
    print(f"  Saved to {MODELS_DIR}")

    rows = []
    for name, info in reg_results.items():
        m = info["test_metrics"]
        rows.append({
            "subset": "FD001",
            "task": "regression",
            "model": name,
            "rmse": m["rmse"],
            "mae": m["mae"],
            "nasa_score": m["nasa_score"],
            "train_seconds": info["train_seconds"],
        })
    for name, info in clf_results.items():
        m = info["test_metrics"]
        rows.append({
            "subset": "FD001",
            "task": f"classification_h{PRIMARY_HORIZON}",
            "model": name,
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "brier": m["brier"],
            "train_seconds": info["train_seconds"],
        })

    print("\n" + "=" * 70)
    print("GENERALIZATION: FD001-tuned XGBoost on other subsets")
    print("=" * 70)
    gen_rows = evaluate_xgb_on_other_subsets(
        xgb_reg_named, xgb_clf_named
    )
    for r in gen_rows:
        r["model"] = "xgboost"
        rows.append(r)

    results_df = pd.DataFrame(rows)
    results_path = TABLES_DIR / "phase2_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults table saved to {results_path}")
    print("\n" + results_df.to_string(index=False))

    params_path = MODELS_DIR / "best_params.json"
    with open(params_path, "w") as f:
        json.dump({
            "xgb_regressor": reg_results["xgboost"]["best_params"],
            "xgb_classifier": clf_results["xgboost"]["best_params"],
        }, f, indent=2)

    print("\n=== Phase 2 training complete ===")


if __name__ == "__main__":
    main()
