from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..cv import make_stratified_folds
from ..data import infer_feature_columns
from ..metrics import auc_per_target


@dataclass
class BaselineConfig:
    n_splits: int
    seed: int
    use_scaler: bool
    C: float
    solver: str
    max_iter: int


def _build_pipeline(num_cols: list[str], cat_cols: list[str], cfg: BaselineConfig) -> Pipeline:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if cfg.use_scaler:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    clf = LogisticRegression(
        C=cfg.C,
        solver=cfg.solver,
        max_iter=cfg.max_iter,
        n_jobs=1,
    )
    return Pipeline(steps=[("prep", preprocessor), ("clf", clf)])


def train_cv_baseline(
    X: pd.DataFrame, y: pd.DataFrame, cfg: BaselineConfig
) -> Tuple[pd.DataFrame, Dict[str, float], list[Pipeline]]:
    num_cols, cat_cols = infer_feature_columns(X)
    oof = pd.DataFrame(index=X.index, columns=y.columns, dtype=float)
    models: list[Pipeline] = []
    for target in y.columns:
        oof[target] = 0.0
    for fold, (tr_idx, va_idx) in enumerate(make_stratified_folds(y, cfg.n_splits, cfg.seed)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        for target in y.columns:
            model = _build_pipeline(num_cols, cat_cols, cfg)
            model.fit(X_tr, y[target].iloc[tr_idx])
            proba = model.predict_proba(X_va)[:, 1]
            oof.loc[X_va.index, target] = proba
            models.append(model)
    scores = auc_per_target(y, oof)
    return oof, scores, models


def predict_baseline(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cfg: BaselineConfig,
) -> pd.DataFrame:
    num_cols, cat_cols = infer_feature_columns(X_train)
    preds = pd.DataFrame(index=X_test.index, columns=y_train.columns, dtype=float)
    for target in y_train.columns:
        model = _build_pipeline(num_cols, cat_cols, cfg)
        model.fit(X_train, y_train[target])
        preds[target] = model.predict_proba(X_test)[:, 1]
    return preds
