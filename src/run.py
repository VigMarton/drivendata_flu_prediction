from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from .blend import BlendConfig, blend_oof_and_test, save_blend_report
from .data import load_submission_format, load_test, load_training
from .eda import run_eda
from .metrics import auc_per_target
from .models.baseline_logreg import BaselineConfig, predict_baseline, train_cv_baseline
from .models.gbdt import GBDTConfig, predict_gbdt, train_cv_gbdt
from .models.nn import NNConfig, train_cv_multitask_nn
from .tuning import TuneConfig, tune_gbdt
from .utils import ensure_dir, read_json, write_json


def _resolve_dirs(config: Dict[str, object]) -> Dict[str, Path]:
    data_dir = Path(config.get("data_dir", "data"))
    runs_dir = Path(config.get("runs_dir", "runs"))
    exp_name = str(config.get("experiment_name", "exp"))
    out_dir = runs_dir / exp_name
    ensure_dir(out_dir)
    return {"data_dir": data_dir, "runs_dir": runs_dir, "out_dir": out_dir}


def _save_predictions(out_dir: Path, name: str, oof: pd.DataFrame, test: pd.DataFrame) -> None:
    oof.to_csv(out_dir / f"oof_{name}.csv", index_label="respondent_id")
    test.to_csv(out_dir / f"test_{name}.csv", index_label="respondent_id")


def _save_submission(out_dir: Path, name: str, data_dir: Path, preds: pd.DataFrame) -> None:
    submission = load_submission_format(data_dir)
    submission[preds.columns] = preds
    submission.to_csv(out_dir / f"submission_{name}.csv", index_label="respondent_id")


def run_baseline(config_path: Path) -> None:
    cfg = read_json(config_path)
    dirs = _resolve_dirs(cfg)
    X, y = load_training(dirs["data_dir"])
    X_test = load_test(dirs["data_dir"])
    model_cfg = BaselineConfig(
        n_splits=int(cfg["n_splits"]),
        seed=int(cfg["seed"]),
        use_scaler=bool(cfg.get("use_scaler", False)),
        C=float(cfg.get("C", 1.0)),
        solver=str(cfg.get("solver", "liblinear")),
        max_iter=int(cfg.get("max_iter", 300)),
    )
    oof, scores, _ = train_cv_baseline(X, y, model_cfg)
    test_preds = predict_baseline(X, y, X_test, model_cfg)
    write_json(dirs["out_dir"] / "baseline_metrics.json", scores)
    _save_predictions(dirs["out_dir"], "baseline", oof, test_preds)
    _save_submission(dirs["out_dir"], "baseline", dirs["data_dir"], test_preds)


def run_eda_cmd(config_path: Path) -> None:
    cfg = read_json(config_path)
    dirs = _resolve_dirs(cfg)
    X, y = load_training(dirs["data_dir"])
    run_eda(X, y, dirs["out_dir"] / "eda")


def run_gbdt_bakeoff(config_path: Path) -> None:
    cfg = read_json(config_path)
    dirs = _resolve_dirs(cfg)
    X, y = load_training(dirs["data_dir"])
    X_test = load_test(dirs["data_dir"])

    results = {}
    for model_type in ["lightgbm", "catboost"]:
        if model_type not in cfg:
            continue
        model_cfg = GBDTConfig(
            n_splits=int(cfg["n_splits"]),
            seed=int(cfg["seed"]),
            model_type=model_type,
            params=cfg[model_type]["params"],
            early_stopping_rounds=int(cfg[model_type].get("early_stopping_rounds", 50)),
        )
        oof, scores = train_cv_gbdt(X, y, model_cfg)
        test_preds = predict_gbdt(X, y, X_test, model_cfg)
        _save_predictions(dirs["out_dir"], model_type, oof, test_preds)
        _save_submission(dirs["out_dir"], model_type, dirs["data_dir"], test_preds)
        results[model_type] = scores

    write_json(dirs["out_dir"] / "gbdt_bakeoff_metrics.json", results)


def run_tuning(config_path: Path) -> None:
    cfg = read_json(config_path)
    dirs = _resolve_dirs(cfg)
    X, y = load_training(dirs["data_dir"])
    tune_cfg = TuneConfig(
        n_trials=int(cfg.get("n_trials", 20)),
        seed=int(cfg.get("seed", 42)),
        model_type=str(cfg["model_type"]),
        early_stopping_rounds=int(cfg.get("early_stopping_rounds", 50)),
    )
    best_params, best_metrics = tune_gbdt(X, y, tune_cfg)
    payload = {"best_params": best_params, "best_metrics": best_metrics}
    write_json(dirs["out_dir"] / "tuning_best.json", payload)


def run_blend(config_path: Path) -> None:
    cfg = read_json(config_path)
    dirs = _resolve_dirs(cfg)
    X, y = load_training(dirs["data_dir"])
    blend_cfg = BlendConfig(
        oof_files=[Path(p) for p in cfg["oof_files"]],
        test_files=[Path(p) for p in cfg["test_files"]],
        n_trials=int(cfg.get("n_trials", 200)),
        seed=int(cfg.get("seed", 42)),
    )
    oof_best, test_best, metrics, weights = blend_oof_and_test(y, blend_cfg)
    _save_predictions(dirs["out_dir"], "blend", oof_best, test_best)
    _save_submission(dirs["out_dir"], "blend", dirs["data_dir"], test_best)
    save_blend_report(dirs["out_dir"], metrics, weights)


def run_multitask_nn(config_path: Path) -> None:
    cfg = read_json(config_path)
    dirs = _resolve_dirs(cfg)
    X, y = load_training(dirs["data_dir"])
    model_cfg = NNConfig(
        n_splits=int(cfg["n_splits"]),
        seed=int(cfg["seed"]),
        epochs=int(cfg.get("epochs", 30)),
        batch_size=int(cfg.get("batch_size", 256)),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        hidden_sizes=[int(h) for h in cfg.get("hidden_sizes", [128, 64])],
        dropout=float(cfg.get("dropout", 0.2)),
        early_stopping_rounds=int(cfg.get("early_stopping_rounds", 5)),
    )
    oof, scores = train_cv_multitask_nn(X, y, model_cfg)
    write_json(dirs["out_dir"] / "nn_metrics.json", scores)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flu Shot Learning experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_eda = subparsers.add_parser("eda", help="Run EDA outputs")
    p_eda.add_argument("--config", required=True, type=Path)

    p_baseline = subparsers.add_parser("baseline", help="Train baseline logistic regression")
    p_baseline.add_argument("--config", required=True, type=Path)

    p_bakeoff = subparsers.add_parser("gbdt_bakeoff", help="Run LightGBM/CatBoost bakeoff")
    p_bakeoff.add_argument("--config", required=True, type=Path)

    p_tune = subparsers.add_parser("tune", help="Tune GBDT hyperparameters")
    p_tune.add_argument("--config", required=True, type=Path)

    p_blend = subparsers.add_parser("blend", help="Blend OOF and test predictions")
    p_blend.add_argument("--config", required=True, type=Path)

    p_nn = subparsers.add_parser("nn", help="Train multitask neural net")
    p_nn.add_argument("--config", required=True, type=Path)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "eda":
        run_eda_cmd(args.config)
    elif args.command == "baseline":
        run_baseline(args.config)
    elif args.command == "gbdt_bakeoff":
        run_gbdt_bakeoff(args.config)
    elif args.command == "tune":
        run_tuning(args.config)
    elif args.command == "blend":
        run_blend(args.config)
    elif args.command == "nn":
        run_multitask_nn(args.config)


if __name__ == "__main__":
    main()
