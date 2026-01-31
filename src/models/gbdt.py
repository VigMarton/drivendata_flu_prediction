from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..cv import make_stratified_folds
from ..data import infer_feature_columns
from ..metrics import auc_per_target


@dataclass
class GBDTConfig:
    n_splits: int
    seed: int
    model_type: str  # "lightgbm" or "catboost"
    params: Dict[str, object]
    early_stopping_rounds: int


def _prepare_cats(X: pd.DataFrame, cat_cols: list[str]) -> Tuple[pd.DataFrame, list[int]]:
    X_copy = X.copy()
    cat_indices = [X_copy.columns.get_loc(c) for c in cat_cols]
    for c in cat_cols:
        X_copy[c] = X_copy[c].astype("category")
    return X_copy, cat_indices


def train_cv_gbdt(
    X: pd.DataFrame, y: pd.DataFrame, cfg: GBDTConfig
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    num_cols, cat_cols = infer_feature_columns(X)
    X_prep, cat_indices = _prepare_cats(X, cat_cols)
    oof = pd.DataFrame(index=X.index, columns=y.columns, dtype=float)
    for target in y.columns:
        oof[target] = 0.0

    for fold, (tr_idx, va_idx) in enumerate(make_stratified_folds(y, cfg.n_splits, cfg.seed)):
        X_tr, X_va = X_prep.iloc[tr_idx], X_prep.iloc[va_idx]
        for target in y.columns:
            y_tr = y[target].iloc[tr_idx]
            y_va = y[target].iloc[va_idx]
            if cfg.model_type == "lightgbm":
                try:
                    import lightgbm as lgb
                except ImportError as exc:
                    raise ImportError("lightgbm is required for model_type=lightgbm") from exc
                model = lgb.LGBMClassifier(**cfg.params)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="auc",
                    categorical_feature=cat_indices,
                    callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
                )
                proba = model.predict_proba(X_va)[:, 1]
            elif cfg.model_type == "catboost":
                try:
                    from catboost import CatBoostClassifier
                except ImportError as exc:
                    raise ImportError("catboost is required for model_type=catboost") from exc
                model = CatBoostClassifier(**cfg.params)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=(X_va, y_va),
                    cat_features=cat_indices,
                    use_best_model=True,
                    verbose=False,
                )
                proba = model.predict_proba(X_va)[:, 1]
            else:
                raise ValueError(f"Unsupported model_type: {cfg.model_type}")
            oof.loc[X_va.index, target] = proba
    scores = auc_per_target(y, oof)
    return oof, scores


def predict_gbdt(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cfg: GBDTConfig,
) -> pd.DataFrame:
    _, cat_cols = infer_feature_columns(X_train)
    X_train_prep, cat_indices = _prepare_cats(X_train, cat_cols)
    X_test_prep, _ = _prepare_cats(X_test, cat_cols)
    preds = pd.DataFrame(index=X_test.index, columns=y_train.columns, dtype=float)

    for target in y_train.columns:
        if cfg.model_type == "lightgbm":
            import lightgbm as lgb

            model = lgb.LGBMClassifier(**cfg.params)
            model.fit(
                X_train_prep,
                y_train[target],
                categorical_feature=cat_indices,
            )
            preds[target] = model.predict_proba(X_test_prep)[:, 1]
        elif cfg.model_type == "catboost":
            from catboost import CatBoostClassifier

            model = CatBoostClassifier(**cfg.params)
            model.fit(
                X_train_prep,
                y_train[target],
                cat_features=cat_indices,
                verbose=False,
            )
            preds[target] = model.predict_proba(X_test_prep)[:, 1]
        else:
            raise ValueError(f"Unsupported model_type: {cfg.model_type}")
    return preds
