from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def make_stratified_folds(
    y: pd.DataFrame, n_splits: int, seed: int
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    if not {"h1n1_vaccine", "seasonal_vaccine"}.issubset(y.columns):
        raise ValueError("y must contain h1n1_vaccine and seasonal_vaccine columns.")
    strat = (2 * y["h1n1_vaccine"].astype(int) + y["seasonal_vaccine"].astype(int)).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return skf.split(y, strat)
