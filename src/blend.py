from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .metrics import auc_per_target
from .utils import read_json, write_json


@dataclass
class BlendConfig:
    oof_files: List[Path]
    test_files: List[Path]
    n_trials: int
    seed: int


def _load_preds(paths: List[Path]) -> List[pd.DataFrame]:
    preds = []
    for p in paths:
        df = pd.read_csv(p, index_col="respondent_id")
        preds.append(df)
    return preds


def blend_oof_and_test(
    y_true: pd.DataFrame, cfg: BlendConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[float]]:
    rng = np.random.RandomState(cfg.seed)
    oof_list = _load_preds(cfg.oof_files)
    test_list = _load_preds(cfg.test_files)
    if len(oof_list) != len(test_list):
        raise ValueError("oof_files and test_files must be the same length.")

    best_weights = None
    best_score = -1.0
    best_metrics: Dict[str, float] = {}
    for _ in range(cfg.n_trials):
        weights = rng.dirichlet(np.ones(len(oof_list)))
        oof_blend = sum(w * df for w, df in zip(weights, oof_list))
        scores = auc_per_target(y_true, oof_blend)
        if scores["mean_auc"] > best_score:
            best_score = scores["mean_auc"]
            best_weights = weights
            best_metrics = scores

    if best_weights is None:
        raise RuntimeError("Failed to compute blend weights.")
    oof_best = sum(w * df for w, df in zip(best_weights, oof_list))
    test_best = sum(w * df for w, df in zip(best_weights, test_list))
    return oof_best, test_best, best_metrics, best_weights.tolist()


def save_blend_report(out_dir: Path, metrics: Dict[str, float], weights: List[float]) -> None:
    payload = {"metrics": metrics, "weights": weights}
    write_json(out_dir / "blend_report.json", payload)
