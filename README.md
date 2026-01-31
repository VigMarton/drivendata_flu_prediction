## Flu Shot Learning rerun

This repo contains a reproducible workflow for the DrivenData Flu Shot Learning competition.

### Quick start
- Place the competition CSVs in `data/` (already provided in this repo).
- Run EDA and baseline:

```
python -m src.run eda --config configs/eda.json
python -m src.run baseline --config configs/baseline_logreg.json
```

The baseline outputs:
- `runs/baseline_logreg/oof_baseline.csv`
- `runs/baseline_logreg/test_baseline.csv`
- `runs/baseline_logreg/submission_baseline.csv`

### GBDT bakeoff
```
python -m src.run gbdt_bakeoff --config configs/gbdt_bakeoff.json
```

Outputs:
- `runs/gbdt_bakeoff/oof_lightgbm.csv`
- `runs/gbdt_bakeoff/oof_catboost.csv`
- `runs/gbdt_bakeoff/submission_lightgbm.csv`
- `runs/gbdt_bakeoff/submission_catboost.csv`

### Hyperparameter tuning
```
python -m src.run tune --config configs/tuning_lgbm.json
python -m src.run tune --config configs/tuning_catboost.json
```

### Blending
Update `configs/blend.json` with the OOF/test paths you want to blend, then:
```
python -m src.run blend --config configs/blend.json
```

### Multitask NN (optional)
```
python -m src.run nn --config configs/nn.json
```

### Notes
- Metric: ROC AUC per label and mean across `h1n1_vaccine` + `seasonal_vaccine`.
- Baseline uses logistic regression with imputation + one-hot encoding.
- GBDT models use LightGBM and CatBoost if installed.
