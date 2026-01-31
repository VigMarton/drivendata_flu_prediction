from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_training(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = pd.read_csv(data_dir / "training_set_features.csv", index_col="respondent_id")
    y = pd.read_csv(data_dir / "training_set_labels.csv", index_col="respondent_id")
    if not X.index.equals(y.index):
        raise ValueError("Training features and labels are misaligned by respondent_id.")
    return X, y


def load_test(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "test_set_features.csv", index_col="respondent_id")


def load_submission_format(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "submission_format.csv", index_col="respondent_id")


def infer_feature_columns(X: pd.DataFrame) -> Tuple[list[str], list[str]]:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols
