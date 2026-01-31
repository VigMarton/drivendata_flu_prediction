from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc_per_target(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for col in y_true.columns:
        scores[col] = roc_auc_score(y_true[col].values, y_pred[col].values)
    scores["mean_auc"] = float(np.mean([scores[c] for c in y_true.columns]))
    return scores
