# Rebalanced Richer 60/20 Condition Analysis

This note compares the earlier adaptive `60/20` benchmark on the baseline simulator against the current adaptive `60/20` benchmark on the rebalanced richer simulator:

- baseline: `analysis/adaptive_60_20/model_comparison.json`
- rebalanced richer: `analysis/adaptive_richer_60_20_rebalanced_full/model_comparison.json`

The rebalanced richer simulator keeps the same bursty, lane-heterogeneous design as the earlier richer profile, but lowers saturation so congestion-style conditions are no longer almost always on. That makes the condition leaderboard more informative. The current main benchmark now also includes the new `neuro_symbolic_tpp` model.

## Current Condition Winners

- `congested`: `neuro_symbolic_tpp` is best with mean balanced accuracy `0.7884`
- `severe_queue`: `neuro_symbolic_tpp` is best with mean balanced accuracy `0.7710`
- `ns_pressure_high`: `neuro_symbolic_tpp` is best with mean balanced accuracy `0.8577`
- `ew_pressure_high`: `neuro_symbolic_tpp` is best with mean balanced accuracy `0.8468`
- `pressure_imbalance`: `neuro_symbolic_tpp` is best with mean balanced accuracy `0.7468`

So the rebalanced richer benchmark now has a clearer condition leader: `neuro_symbolic_tpp` is strongest overall, while `transformer_tpp` and `multitask_neural_tpp` remain the best purely neural competitors.

## Full Learned-Model Readout

- `neuro_symbolic_tpp`
  - mean condition balanced accuracy: `0.8021`
  - mean Brier: `0.1076`
  - mean log loss: `0.3369`
- `transformer_tpp`
  - `0.7901`
  - `0.1131`
  - `0.3566`
- `multitask_neural_tpp`
  - `0.7788`
  - `0.1190`
  - `0.3731`
- `continuous_tpp`
  - `0.7677`
  - `0.1256`
  - `0.3931`
- `neural_tpp`
  - `0.6642`
  - `0.3672`
  - `5.0727`

The direct-condition models still dominate the rollout-derived `neural_tpp`, and the strongest model is now the neuro-symbolic variant rather than a purely neural architecture.

## What Changed Relative To The Earlier Richer Run

The earlier richer profile was too saturated: congestion and severe-queue targets were positive almost all the time. After the simulator rebalance:

- the direct-condition models are closer together than in the baseline simulator
- the neuro-symbolic branch pushes condition quality higher again by injecting explicit rule structure
- `continuous_tpp` recovers a healthier middle position instead of collapsing on condition quality
- `multitask_neural_tpp` remains strong, but is no longer the default best model on every congestion-like target

This suggests the old richer profile was rewarding persistence handling more than genuinely discriminative condition forecasting, and that explicit symbolic condition logic is especially effective once the simulator becomes less saturated.

## Event And Condition Specialization

The rebalanced richer run makes model specialization clearer:

- best exact type accuracy: `neural_tpp` at `0.4915`
- best family accuracy: `neuro_symbolic_tpp` at `0.6290`
- best timing: `continuous_tpp` with time MAE `0.4473`
- best condition forecasting: `neuro_symbolic_tpp` at `0.8021`

That split is useful. We no longer have one model winning every task slice. Instead:

- `neural_tpp` is still best at exact next-event typing
- `continuous_tpp` is best at timing
- `neuro_symbolic_tpp` is best at fixed-horizon condition risk and also strongest on family accuracy among the learned models

## Per-Condition Interpretation

### `neuro_symbolic_tpp`

This is now the strongest richer-simulator condition model overall.

- best on all five pooled condition slices
- best pooled probabilistic quality too
- likely benefiting from combining explicit rule-threshold structure with a learned event-history residual, which is a natural fit for benchmark conditions like congestion and pressure imbalance

### `transformer_tpp`

Still the strongest purely neural condition model.

- second overall on pooled condition metrics
- still a very strong option if we want a model with no explicit symbolic branch
- likely benefiting from wider context access now that the simulator produces more varied buildup and recovery patterns

### `multitask_neural_tpp`

Still a very strong direct-condition specialist.

- third overall on pooled condition metrics
- remains robust across all five conditions
- now clearly behind the neuro-symbolic model on both balanced accuracy and calibration-sensitive scores

### `continuous_tpp`

Still the strongest timing-oriented model.

- best one-step family accuracy and time MAE
- condition forecasting improved substantially over the earlier oversaturated richer setup
- still trails the top two direct-condition specialists on pooled condition quality

### `neural_tpp`

Still strongest on exact next-event type accuracy.

- good event model
- much weaker condition model
- still badly behind on probabilistic condition quality because conditions are derived indirectly from rollout

## Main Takeaways

- The simulator rebalance was worth it.
- The richer benchmark is now less saturated and more diagnostic.
- `neuro_symbolic_tpp` is now the best overall condition forecaster on the main rebalanced richer full benchmark.
- `transformer_tpp` remains the strongest purely neural condition model.
- `continuous_tpp` remains the best timing model.
- `neural_tpp` remains best on exact next-event type accuracy, which keeps the event-versus-condition split very clear.

## Recommended Next Model Work

1. Compare `neuro_symbolic_tpp` and `transformer_tpp` more deeply on calibration and per-condition failure cases.
2. Keep `continuous_tpp` as the main timing specialist rather than forcing it to win the condition task too.
3. Expand the simulator again with turning movements and spillback-style constraints before introducing another new model family.
