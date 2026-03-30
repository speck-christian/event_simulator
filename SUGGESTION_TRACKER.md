# Suggestion Tracker

This file tracks major modeling and evaluation suggestions from the project conversation, along with whether they have been executed in the repo yet.

## Executed

- Adaptive signal controller:
  Phase timing is now influenced by queue size and arrival-rate pressure in the traffic simulator.

- Per-family event metrics:
  One-step and rollout event metrics are broken out by `phase_change`, `vehicle_arrival`, and `vehicle_departure`.

- Browser dashboard improvements:
  The comparison dashboard includes interactive legends, model filtering, per-family plots, and condition-forecast plots.

- Event-derived condition forecasting:
  Higher-level conditions such as congestion and corridor pressure are computed from predicted rollouts.

- Fixed-time condition evaluation:
  The evaluation pipeline now scores conditions at `10s`, `30s`, and `60s` future horizons.

- Revised condition metrics:
  Condition plots now use `balanced_accuracy`, and the JSON output also includes `f1`, `precision`, and `recall`.

- Colab/GPU preparation:
  Learned models accept a `--device` argument, and the repo includes a Colab-oriented learned-model training entrypoint and starter notebook.

- Direct fixed-horizon condition prediction:
  Added `multitask_neural_tpp`, a GRU-based learned model with direct condition heads for fixed-time condition forecasting.

- Initial multitask tuning pass:
  Added lower condition-loss weighting, imbalance-aware condition loss, gradient clipping, and learned per-condition thresholds for `multitask_neural_tpp`.

- Handcrafted condition-feature augmentation:
  Added explicit corridor-pressure aggregate features to the multitask condition model input to improve fixed-horizon condition forecasting.

- Adaptive-aware mechanistic baseline:
  The mechanistic baseline now uses adaptive green-control timing logic, better arrival timing, and probabilistic time-conditioned condition forecasts.

- Direct condition heads for `continuous_tpp`:
  The continuous-time LSTM baseline now includes direct fixed-horizon condition heads, learned thresholds, and probability outputs for condition evaluation.

- Probabilistic condition evaluation:
  Time-conditioned condition metrics now include probability-aware scores such as Brier score and log loss, and the dashboard includes an aggregate probabilistic condition section.

- Direct condition heads for `transformer_tpp`:
  The attention-based TPP baseline now includes direct fixed-horizon condition heads, learned thresholds, and probability outputs for condition evaluation.

- Calibration plots:
  The dashboard now includes pooled reliability-style calibration plots for time-conditioned condition probabilities.

- Validation-based threshold tuning:
  Threshold tuning is now model-specific: `continuous_tpp` uses held-out whole-run validation splits, while `multitask_neural_tpp` and `transformer_tpp` use the stronger full-sample threshold tuning path.

- Post-hoc condition calibration:
  The direct-condition learned models now apply per-condition Platt-style post-hoc calibration before scoring and thresholding their fixed-time condition probabilities, and the dashboard includes per-condition probabilistic metric plots.

- Per-condition calibration diagnostics:
  Evaluation output now includes reliability bins for each individual fixed-time condition, and the dashboard renders per-condition calibration cards in addition to the pooled reliability view.

- Per-condition richer-simulator analysis:
  The repo now includes a focused comparison of baseline-vs-richer adaptive `60/20` condition behavior, including which condition targets became harder and which models remained strongest.

- Richer-simulator loss-reweighting experiment:
  Additional richer-profile metadata features and manual condition-target loss weights were tried for `multitask_neural_tpp` and `continuous_tpp`, but the full richer `60/20` evaluation regressed, so that tuning pass was rolled back.

- Calibration-only threshold refinement:
  The post-hoc condition threshold search now uses a denser, data-driven candidate set derived from calibrated probabilities instead of only a coarse fixed grid. On the richer `60/20` rerun, this clearly improved `multitask_neural_tpp` condition balanced accuracy, but did not improve `continuous_tpp`, so `continuous_tpp` was kept on the earlier coarse-grid threshold path.

- Transformer condition-path refinement:
  `transformer_tpp` now uses a dedicated condition trunk that fuses the transformer event trunk with raw state features before condition prediction. On the richer adaptive `60/20` rerun, this materially improved condition balanced accuracy and probabilistic condition quality.

- Transformer loss-weight refinement:
  Reducing the transformer condition-loss weight from `0.10` to `0.08` recovered event-family accuracy on the richer adaptive `60/20` benchmark while preserving the condition-forecast gains from the dedicated condition trunk.

- Learned-model checkpointing:
  Learned-model training now saves reloadable checkpoints, vocab state, calibration artifacts, and thresholds under the benchmark output directory so later eval-only and dashboard/report work do not require retraining from scratch.

- Long-run progress logging:
  The training and evaluation CLIs now print fit/eval progress per model and per evaluation run, which makes long local or Colab runs much easier to monitor.

- Rebalanced richer simulator:
  The richer simulator profile was retuned to reduce persistent overload by lowering base demand, softening burstiness, slightly improving green-phase service, and making adaptive control more responsive to current demand. This keeps the richer profile realistic without making congestion-style labels almost always positive.

- Rebalanced richer full benchmark:
  The main adaptive richer `60/20` benchmark has been rerun locally after the simulator rebalance, with checkpoints and refreshed dashboards written under `analysis/adaptive_richer_60_20_rebalanced_full`. In that current benchmark, `transformer_tpp` is the strongest overall condition forecaster, `continuous_tpp` is strongest on timing and family accuracy, and `neural_tpp` is strongest on exact type accuracy.

- Neuro-symbolic TPP prototype:
  Added `neuro_symbolic_tpp`, a rule-guided GRU temporal point process with an explicit symbolic condition-logic branch plus a neural residual head. A focused smoke benchmark under `analysis/neuro_symbolic_smoke` ran end to end with checkpoints and showed promising condition performance, outperforming the smoke-run `multitask_neural_tpp` and `transformer_tpp` on mean fixed-horizon condition balanced accuracy.

- Neuro-symbolic full benchmark promotion:
  The full adaptive richer `60/20` run for `neuro_symbolic_tpp` completed under `analysis/adaptive_richer_60_20_neuro_symbolic_only`, and the main benchmark comparison/report artifacts were updated so `neuro_symbolic_tpp` is included in the current leaderboard.

## Pending

- Probability-aware condition evaluation:
  Brier score, log loss, pooled calibration plots, per-condition probabilistic plots, and per-condition calibration diagnostics are implemented, but PR-AUC and richer calibration-error summaries are still unimplemented.

- Direct state forecasting:
  There is not yet a model trained to forecast future queue/state variables directly rather than via events or condition heads.

- Stronger rollout-aware training:
  The transformer has a rollout-aware training path, but longer-horizon event drift is still an open improvement area.

- Stronger temporal point process architectures:
  More faithful neural Hawkes-style intensity models remain unimplemented.

- Flow matching experiments:
  Flow matching has been discussed conceptually, but there is no flow-matching model or evaluation path in the repo yet.

- Cross-domain application spaces:
  The codebase and report are still centered on traffic simulation; additional application spaces have not been added yet.
