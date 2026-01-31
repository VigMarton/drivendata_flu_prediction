from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from .data import infer_feature_columns
from .utils import ensure_dir


def _safe_codes(series: pd.Series) -> np.ndarray:
    filled = series.fillna("__MISSING__").astype("category")
    return filled.cat.codes.to_numpy()


def run_eda(X: pd.DataFrame, y: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)

    missing = X.isna().mean().sort_values(ascending=False)
    missing.to_frame("missing_rate").to_csv(out_dir / "missingness.csv")

    _, cat_cols = infer_feature_columns(X)
    cat_cardinality = (
        X[cat_cols].nunique(dropna=False).sort_values(ascending=False)
        if cat_cols
        else pd.Series(dtype=int)
    )
    cat_cardinality.to_frame("cardinality").to_csv(out_dir / "categorical_cardinality.csv")

    assoc_rows = []
    for target in y.columns:
        y_target = y[target].values
        num_cols, cat_cols = infer_feature_columns(X)
        if num_cols:
            num_data = X[num_cols].to_numpy()
            num_data = np.nan_to_num(num_data, nan=np.nanmean(num_data, axis=0))
            mi_num = mutual_info_classif(num_data, y_target, discrete_features=False)
            assoc_rows.extend(
                [{"target": target, "feature": col, "mi": float(mi)} for col, mi in zip(num_cols, mi_num)]
            )
        if cat_cols:
            cat_data = np.stack([_safe_codes(X[c]) for c in cat_cols], axis=1)
            mi_cat = mutual_info_classif(cat_data, y_target, discrete_features=True)
            assoc_rows.extend(
                [{"target": target, "feature": col, "mi": float(mi)} for col, mi in zip(cat_cols, mi_cat)]
            )

    assoc_df = pd.DataFrame(assoc_rows).sort_values(["target", "mi"], ascending=[True, False])
    assoc_df.to_csv(out_dir / "feature_association_mi.csv", index=False)
