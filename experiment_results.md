# Experiment Results (Completed)

This document summarizes completed experiments, their results, and conclusions so far.

## Baselines and Bakeoff

### Logistic Regression baseline
- **Metric (CV mean AUC)**: `0.84364`
- **Details**: `h1n1=0.83366`, `seasonal=0.85363`
- **Conclusion**: Useful baseline; substantially behind GBDT models.

### GBDT bakeoff (default configs)
- **LightGBM**: mean AUC `0.86662` (`h1n1=0.86935`, `seasonal=0.86389`)
- **CatBoost**: mean AUC `0.86808` (`h1n1=0.87121`, `seasonal=0.86494`)
- **Conclusion**: CatBoost slightly ahead of LightGBM in this bakeoff.

### Blend (LightGBM + CatBoost)
- **Metric (CV mean AUC)**: `0.86907`
- **Details**: `h1n1=0.87219`, `seasonal=0.86596`
- **Weights**: LightGBM `0.3886`, CatBoost `0.6114`
- **Conclusion**: Best overall result to date among baseline/bakeoff/blend.

## Feature Engineering (Survey Structure)
- **Base LGBM (notebook)**: mean AUC `0.85497`
- **With engineered features**: mean AUC `0.85577`
- **Gain**: `+0.00080`
- **Features added**: `behavioral_sum`, `opinion_effective_gap`, `opinion_risk_gap`,
  `opinion_sick_gap`, `doctor_recc_any`
- **Conclusion**: Small but consistent lift; keep features for later combinations.

## Pseudo‑labeling (Holdout Simulation)
- **Baseline holdout (no pseudo‑labels)**: mean AUC `0.85497`
- **Pseudo‑labels @ 0.98 threshold**: mean AUC `0.85587`
- **Pseudo‑labels @ 0.90 threshold**: mean AUC `0.85587`
- **Gain**: `+0.00090` (no improvement from lower threshold)
- **Conclusion**: Slight lift; not clearly worth the added complexity.

## Stacking / Meta‑Learner Experiments
- **Meta‑only stacking (OOF preds)**: mean AUC `0.86799`
- **Meta + one‑hot features**: mean AUC `0.86644`
- **Meta + LOO encoding**: mean AUC `0.86648` (removed)
- **CatBoost meta‑learner (OOF‑only)**: mean AUC `0.867243`
- **Conclusion**: None beat the simple blend (`0.86907`).

## Segmented Modeling (Mixture‑of‑Experts)
Segmenting uses a global LightGBM plus segment‑specific models blended per row.

- **`age_group`**: mean AUC `0.85408` (worse than base)
- **`health_worker`**: mean AUC `0.85639` (best of the three; small lift)
- **`doctor_recc_any`**: mean AUC `0.85537` (small lift, weaker than `health_worker`)
- **Conclusion**: Only `health_worker` looks worth carrying forward; gains are modest.

## Calibration‑Aware Blending
- **Base blend**: mean AUC `0.869073`
- **Sigmoid‑calibrated blend**: mean AUC `0.869079` (no meaningful change)
- **Isotonic‑calibrated blend**: mean AUC `0.869758` (**+0.000684**)
- **Conclusion**: Isotonic calibration improves the blend and is the current best.

## Rule‑Augmented Post‑Processing
- **Base blend**: mean AUC `0.869073`
- **Doctor‑only rule best**: mean AUC `0.869075` (delta `+0.000002`)
- **Doctor + health_worker best**: mean AUC `0.869086` (delta `+0.000012`)
- **Conclusion**: Gains are negligible; discard.

## Block / Row‑Pattern Analysis
- **Unsupervised breakpoints**: no stable block regimes beyond the train/test boundary.
- **Label impact by detected blocks**: deltas within ±0.004 across blocks.
- **Conclusion**: No meaningful block‑specific effect; discard.

## Distribution Shift Checks
- **Train vs test drift AUC**: `0.5014` (near random).
- **Top feature shifts**: all small (largest shift_score ~0.013).
- **Conclusion**: No actionable drift; CV likely representative.

## Overall Best So Far
- **Best CV score**: Isotonic‑calibrated blend at **mean AUC `0.869758`**
- **Promising add‑ons**: survey‑structure features, `health_worker` segmentation (modest)

