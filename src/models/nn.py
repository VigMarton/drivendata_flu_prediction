from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..cv import make_stratified_folds
from ..data import infer_feature_columns
from ..metrics import auc_per_target


@dataclass
class NNConfig:
    n_splits: int
    seed: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden_sizes: List[int]
    dropout: float
    early_stopping_rounds: int


def _prepare_tabular(
    X: pd.DataFrame, num_cols: list[str], cat_cols: list[str]
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    X_num = X[num_cols].to_numpy(dtype=np.float32)
    X_num = np.nan_to_num(X_num, nan=np.nanmean(X_num, axis=0))

    cat_cardinalities = []
    cat_arrays = []
    for col in cat_cols:
        series = X[col].fillna("__MISSING__").astype("category")
        cat_cardinalities.append(series.cat.categories.size + 1)
        cat_arrays.append(series.cat.codes.to_numpy(dtype=np.int64))
    X_cat = np.stack(cat_arrays, axis=1) if cat_arrays else np.zeros((len(X), 0), dtype=np.int64)
    return X_num, X_cat, cat_cardinalities


def train_cv_multitask_nn(
    X: pd.DataFrame, y: pd.DataFrame, cfg: NNConfig
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("torch is required for the multitask NN approach") from exc

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    num_cols, cat_cols = infer_feature_columns(X)
    X_num, X_cat, cat_cards = _prepare_tabular(X, num_cols, cat_cols)
    y_np = y.to_numpy(dtype=np.float32)

    class TabularNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            emb_layers = []
            emb_dim_total = 0
            for card in cat_cards:
                dim = min(50, (card + 1) // 2)
                emb_layers.append(nn.Embedding(card, dim))
                emb_dim_total += dim
            self.embeddings = nn.ModuleList(emb_layers)
            input_dim = emb_dim_total + X_num.shape[1]
            layers: List[nn.Module] = []
            for hidden in cfg.hidden_sizes:
                layers.append(nn.Linear(input_dim, hidden))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(cfg.dropout))
                input_dim = hidden
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(input_dim, 2)

        def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
            if self.embeddings:
                emb = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
                emb = torch.cat(emb, dim=1)
                x = torch.cat([x_num, emb], dim=1)
            else:
                x = x_num
            feats = self.backbone(x)
            return self.head(feats)

    oof = pd.DataFrame(index=X.index, columns=y.columns, dtype=float)
    for target in y.columns:
        oof[target] = 0.0

    for fold, (tr_idx, va_idx) in enumerate(make_stratified_folds(y, cfg.n_splits, cfg.seed)):
        X_tr_num, X_va_num = X_num[tr_idx], X_num[va_idx]
        X_tr_cat, X_va_cat = X_cat[tr_idx], X_cat[va_idx]
        y_tr, y_va = y_np[tr_idx], y_np[va_idx]

        model = TabularNN()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        train_ds = TensorDataset(
            torch.tensor(X_tr_num), torch.tensor(X_tr_cat), torch.tensor(y_tr)
        )
        val_ds = TensorDataset(
            torch.tensor(X_va_num), torch.tensor(X_va_cat), torch.tensor(y_va)
        )
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        best_auc = -1.0
        patience = 0
        for epoch in range(cfg.epochs):
            model.train()
            for xb_num, xb_cat, yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb_num, xb_cat)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            preds = []
            with torch.no_grad():
                for xb_num, xb_cat, _ in val_loader:
                    logits = model(xb_num, xb_cat)
                    preds.append(torch.sigmoid(logits).cpu().numpy())
            preds = np.vstack(preds)
            oof.loc[X.iloc[va_idx].index, :] = preds
            scores = auc_per_target(y.iloc[va_idx], oof.loc[X.iloc[va_idx].index])
            if scores["mean_auc"] > best_auc:
                best_auc = scores["mean_auc"]
                patience = 0
            else:
                patience += 1
                if patience >= cfg.early_stopping_rounds:
                    break
    scores = auc_per_target(y, oof)
    return oof, scores
