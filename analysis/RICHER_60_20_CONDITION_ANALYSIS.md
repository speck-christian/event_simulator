# Richer 60/20 Condition Analysis

This note compares the earlier adaptive `60/20` benchmark on the baseline simulator against the adaptive `60/20` benchmark on the richer simulator:

- baseline: `analysis/adaptive_60_20/model_comparison.json`
- richer: `analysis/adaptive_richer_60_20/model_comparison.json`

The richer simulator introduces:

- time-varying corridor demand pulses
- bursty local arrivals
- lane-specific service headways

The main question here is not only whether scores change, but which condition targets change and which models remain strongest under the richer environment.

## Condition Winners On Richer 60/20

- `congested`: `multitask_neural_tpp` remains best with mean balanced accuracy `0.7530`
- `severe_queue`: `multitask_neural_tpp` remains best with mean balanced accuracy `0.7559`
- `ns_pressure_high`: `multitask_neural_tpp` remains best with mean balanced accuracy `0.8095`
- `ew_pressure_high`: `multitask_neural_tpp` remains best with mean balanced accuracy `0.8685`
- `pressure_imbalance`: `continuous_tpp` is best with mean balanced accuracy `0.8042`, narrowly above `multitask_neural_tpp` at `0.8023`

So the richer benchmark does not overturn the general condition-forecast story, but it does make the leaderboard less uniform. In particular, `continuous_tpp` becomes the strongest model on imbalance forecasting.

## Which Conditions Got Harder

The richer simulator makes some condition targets much harder for the learned direct-condition models:

- `congested`
  - `continuous_tpp`: `0.7520 -> 0.5902`
  - `transformer_tpp`: `0.7997 -> 0.6736`
  - `neural_tpp`: `0.7564 -> 0.6678`
- `severe_queue`
  - `continuous_tpp`: `0.7300 -> 0.6006`
  - `neural_tpp`: `0.7571 -> 0.6161`
  - `transformer_tpp`: `0.7976 -> 0.7164`
- `ns_pressure_high`
  - `continuous_tpp`: `0.8788 -> 0.6559`
  - `transformer_tpp`: `0.8121 -> 0.7064`
  - `multitask_neural_tpp`: `0.8703 -> 0.8095`

These are the clearest signs that the richer simulator is exposing new structural difficulty rather than merely injecting noise.

## Which Conditions Became More Distinctive

Some condition targets actually become easier or more separable for certain models:

- `ew_pressure_high`
  - `transition`: `0.5221 -> 0.7146`
  - `neural_tpp`: `0.6319 -> 0.7497`
  - `continuous_tpp`: `0.8566 -> 0.8627`
- `pressure_imbalance`
  - `mechanistic`: `0.5041 -> 0.6998`
  - `continuous_tpp`: `0.7225 -> 0.8042`
  - `multitask_neural_tpp`: `0.7450 -> 0.8023`

This is consistent with the richer simulator adding stronger corridor-specific and lane-specific asymmetries. The new environment makes directional pressure more informative, especially for east-west pressure and imbalance.

## Model-Specific Readout

### `multitask_neural_tpp`

Still the best overall condition forecaster.

- strongest on `congested`, `severe_queue`, `ns_pressure_high`, and `ew_pressure_high`
- drops under richer dynamics, but less sharply than `continuous_tpp` and `transformer_tpp`
- remains the most robust direct-condition model overall

Suggested next step:

- improve short-horizon congestion and severe-queue sensitivity under bursty arrivals

### `continuous_tpp`

Still strongest on one-step timing, but more condition-sensitive under the richer simulator.

- takes over first place on `pressure_imbalance`
- degrades sharply on `congested`, `severe_queue`, and especially `ns_pressure_high`
- likely benefiting from its time representation, but losing some of the direct-condition edge under nonstationary demand

Suggested next step:

- strengthen direct condition supervision for congestion-like conditions rather than pressure-asymmetry conditions

### `transformer_tpp`

Remains competitive, but loses more than `multitask_neural_tpp` on several richer condition targets.

- still second on `severe_queue` and `ns_pressure_high`
- loses ground on `congested`
- retains strong `ew_pressure_high` behavior

Suggested next step:

- improve robustness to bursty local demand rather than just broader context aggregation

### `neural_tpp`

The rollout-derived condition model is still weaker than the direct-condition models.

- improves on `ew_pressure_high`
- loses ground on the broader congestion-style conditions
- remains much less calibrated than the direct-condition models, even where thresholded balanced accuracy is acceptable

Suggested next step:

- no new tuning priority here unless event-only forecasting becomes the primary target again

### `mechanistic`

Becomes more interpretable under the richer simulator.

- much better on `pressure_imbalance`
- slightly better on `ns_pressure_high`
- worse on `congested` and `severe_queue`

This suggests the mechanistic model is good at coarse directional structure, but weaker at capturing exact queue growth under bursty arrivals and lane-specific service bottlenecks.

Suggested next step:

- explicitly model nonstationary arrival pulses if we want this baseline to stay competitive on congestion-style conditions

## Main Takeaways

- The richer simulator does distinguish the models more meaningfully than the earlier baseline simulator.
- The direct-condition learned models still lead overall, so the main modeling conclusion survives the richer environment.
- The condition tasks are no longer uniformly hard:
  - congestion-style conditions got harder
  - corridor-asymmetry conditions became more structured
- `multitask_neural_tpp` is now the clearest “best default” condition model.
- `continuous_tpp` has a more specialized niche: strongest timing model, and now also strongest on `pressure_imbalance`.

## Recommended Next Model Work

1. Improve `multitask_neural_tpp` on short-horizon `congested` and `severe_queue`.
2. Improve `continuous_tpp` on `ns_pressure_high` and congestion-style condition heads.
3. Use `pressure_imbalance` as a targeted evaluation slice when tuning `continuous_tpp`, since that is where it now clearly adds value.
