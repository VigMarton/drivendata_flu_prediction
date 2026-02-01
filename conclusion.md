# Conclusion

This document summarizes what worked, what did not, and the overall impact of AI
collaboration on the workflow and results for the H1N1 competition rerun.

## What Worked (and Why)
- **LightGBM + CatBoost blending**: Strongest and most stable gains. The two models
  capture slightly different patterns and blend well.
- **Isotonic calibration on the blend**: Small but consistent improvement by
  re-mapping probabilities to better align with true ranking.
- **Survey-structure features**: Simple, domain-aligned aggregates and gaps added
  a small lift and carried through to the best blend.
- **Systematic experiment pipeline**: Fast iteration, reusable configs, and clear
  experiment logs enabled quick convergence on the best approach.

## What Did Not Work (and Why)
- **Pseudo-labeling**: Only marginal improvements in holdout simulations; likely
  too much noise for the benefit at this signal level.
- **Segmented modeling**: Small gains at best; segment-specific models did not
  generalize strongly enough to justify complexity.
- **Stacking / meta learners**: None surpassed a simple blend; meta models likely
  overfit or added noise.
- **Rule-augmented post-processing**: Effects were negligible and within noise.
- **Block/index pattern analysis**: No stable block regimes or label shifts; no
  meaningful signal to exploit.
- **Distribution shift adjustments**: Train/test drift was effectively random,
  so no adjustment was justified.
- **Custom NN model**: Underperformed even the logistic baseline; tabular GBDT
  remains the best fit for this dataset.

## Final Result
- **Best submission score**: `0.8631`
- **Leaderboard placement**: ~150th out of 8000+ teams/individuals
- **Comparison**: Slightly behind the solo run (22nd place), but achieved in
  significantly less time and effort.

## AI Collaboration Impact
- **Workflow acceleration**: Project setup, experiment scaffolding, and repeated
  testing were much faster with AI support.
- **Parallel experimentation**: Multiple approaches were tested quickly and
  sequentially without long idle time.
- **Idea overlap**: AI suggested approaches largely aligned with the user’s solo
  strategy, which is expected for a well-studied, simple ML competition.
- **Net effect**: The final score was achieved rapidly, confirming that AI can
  compress time-to-result even when absolute gains are small.

## Why Improvements Plateaued
- The problem is **simple and well-explored**, so top methods converge quickly.
- **Hyperparameter tuning, preprocessing, and loss tweaks** deliver diminishing
  returns once strong GBDTs and blending are in place.
- At this stage, **variance and randomness** (e.g., seed choice) can dominate
  tiny AUC deltas.

## What Could Improve the Score (Not Yet Tried)
- **Multi-seed ensembling**: Train the same LGBM/CatBoost configs on multiple
  seeds and average predictions (often the most reliable late-stage gain).
- **XGBoost as a third blender**: Add XGBoost to the blend for diversity.
- **Heavy feature crosses**: Targeted interaction features (e.g., risk × age).
- **Monotonic constraints**: Enforce monotonicity for opinion/risk features.
- **More extensive calibration search**: Compare isotonic vs spline or
  beta-calibration methods for ranking improvements.

