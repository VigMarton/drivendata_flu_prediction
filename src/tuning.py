from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .cv import make_stratified_folds
from .data import infer_feature_columns
from .metrics import auc_per_target


@dataclass
class TuneConfig:
    n_trials: int
    seed: int
    model_type: str
    early_stopping_rounds: int


def _sample_lightgbm(rng: np.random.RandomState) -> Dict[str, object]:
    return {
        "n_estimators": int(rng.randint(300, 1200)),
        "learning_rate": float(rng.uniform(0.01, 0.2)),
        "num_leaves": int(rng.randint(16, 128)),
        "max_depth": int(rng.randint(3, 10)),
        "min_data_in_leaf": int(rng.randint(20, 200)),
        "feature_fraction": float(rng.uniform(0.6, 1.0)),
        "bagging_fraction": float(rng.uniform(0.6, 1.0)),
        "bagging_freq": int(rng.randint(1, 8)),
        "lambda_l1": float(rng.uniform(0.0, 1.0)),
        "lambda_l2": float(rng.uniform(0.0, 1.0)),
        "objective": "binary",
        "n_jobs": 4,
    }


def _sample_catboost(rng: np.random.RandomState) -> Dict[str, object]:
    return {
        "iterations": int(rng.randint(300, 1200)),
        "learning_rate": float(rng.uniform(0.01, 0.2)),
        "depth": int(rng.randint(4, 10)),
        "l2_leaf_reg": float(rng.uniform(1.0, 10.0)),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
    }


def _prepare_cats(X: pd.DataFrame, cat_cols: list[str]) -> Tuple[pd.DataFrame, list[int]]:
    X_copy = X.copy()
    cat_indices = [X_copy.columns.get_loc(c) for c in cat_cols]
    for c in cat_cols:
        X_copy[c] = X_copy[c].astype("category")
    return X_copy, cat_indices


def tune_gbdt(
    X: pd.DataFrame, y: pd.DataFrame, cfg: TuneConfig
) -> Tuple[Dict[str, object], Dict[str, float]]:
    rng = np.random.RandomState(cfg.seed)
    best_params: Dict[str, object] = {}
    best_score = -1.0
    best_metrics: Dict[str, float] = {}

    num_cols, cat_cols = infer_feature_columns(X)
    X_prep, cat_indices = _prepare_cats(X, cat_cols)

    for _ in range(cfg.n_trials):
        if cfg.model_type == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError as exc:
                raise ImportError("lightgbm is required for tuning") from exc
            params = _sample_lightgbm(rng)
            oof = pd.DataFrame(index=X.index, columns=y.columns, dtype=float)
            for target in y.columns:
                oof[target] = 0.0
            for tr_idx, va_idx in make_stratified_folds(y, 3, cfg.seed):
                X_tr, X_va = X_prep.iloc[tr_idx], X_prep.iloc[va_idx]
                for target in y.columns:
                    y_tr = y[target].iloc[tr_idx]
                    y_va = y[target].iloc[va_idx]
                    model = lgb.LGBMClassifier(**params)
                    model.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_va, y_va)],
                        eval_metric="auc",
                        categorical_feature=cat_indices,
                        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
                    )
                    oof.loc[X_va.index, target] = model.predict_proba(X_va)[:, 1]
        elif cfg.model_type == "catboost":
            try:
                from catboost import CatBoostClassifier
            except ImportError as exc:
                raise ImportError("catboost is required for tuning") from exc
            params = _sample_catboost(rng)
            oof = pd.DataFrame(index=X.index, columns=y.columns, dtype=float)
            for target in y.columns:
                oof[target] = 0.0
            for tr_idx, va_idx in make_stratified_folds(y, 3, cfg.seed):
                X_tr, X_va = X_prep.iloc[tr_idx], X_prep.iloc[va_idx]
                for target in y.columns:
                    y_tr = y[target].iloc[tr_idx]
                    y_va = y[target].iloc[va_idx]
                    model = CatBoostClassifier(**params)
                    model.fit(
                        X_tr,
                        y_tr,
                        eval_set=(X_va, y_va),
                        cat_features=cat_indices,
                        use_best_model=True,
                        verbose=False,
                    )
                    oof.loc[X_va.index, target] = model.predict_proba(X_va)[:, 1]
        else:
            raise ValueError(f"Unsupported model_type: {cfg.model_type}")

        scores = auc_per_target(y, oof)
        if scores["mean_auc"] > best_score:
            best_score = scores["mean_auc"]
            best_params = params
            best_metrics = scores
    return best_params, best_metrics
