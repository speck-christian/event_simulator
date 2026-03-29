# Event Prediction Modeling Approaches

This note frames a practical model ladder for predicting future events in the traffic-intersection simulator.

## What to optimize for first

Before chasing the most expressive model family, it is worth separating three use cases:

- next-event prediction
  - predict the next event type and its time
- short-horizon rollout
  - predict the next several events over the next few seconds
- long-horizon joint generation
  - generate an entire plausible future event trace

Those are related, but not identical. Flow Matching is most attractive for the third case. For the current simulator, I would start by getting excellent at the first two.

## Recommended baseline ladder

### 1. Mechanistic baseline

Use the known controller structure and queue state directly.

- inputs
  - current phase
  - time remaining in phase
  - per-lane queue length
  - estimated arrival rates
- strengths
  - strong on structured systems with known rules
  - transparent failures
  - great sanity-check for any learned model
- weakness
  - brittle if the simulator dynamics become richer or partially observed

### 2. Empirical transition baseline

Learn the most likely next event and delay from historical transitions.

- strengths
  - simple
  - easy to debug
  - useful lower bound for learned models
- weakness
  - weak under longer context or distribution shift

### 3. Global rate / Poisson-style baseline

Use event recurrence statistics without much context.

- strengths
  - tiny implementation cost
  - useful as a floor
- weakness
  - ignores queue and controller interactions

## Learned model families worth considering

### Neural temporal point processes

This is the most natural first learned family for continuous-time event prediction.

- good fit when you want to predict both `what happens next` and `when`
- can model marked events directly
- easier to evaluate on likelihood and next-event metrics than full generative sequence models

Two canonical references:

- Neural Hawkes Process
  - https://arxiv.org/abs/1612.09328
- Transformer Hawkes Process
  - https://arxiv.org/abs/2002.09291

Why they matter here:

- Neural Hawkes uses a continuous-time recurrent state, which matches asynchronous event logs well.
- Transformer Hawkes replaces recurrence with attention, which can help when longer context matters more than local recency.

### Intensity-free / flow-based temporal point process models

These model the event-sequence distribution more directly instead of focusing only on intensities.

Reference:

- Point Process Flows
  - https://arxiv.org/abs/1910.08281

Why it may help:

- useful when the event-time distribution is complicated or hard to capture with a hand-designed intensity
- closer in spirit to direct distribution modeling than classic TPPs

### Flow Matching

Reference:

- Flow Matching for Generative Modeling
  - https://arxiv.org/abs/2210.02747

Why it is interesting:

- strong for learning joint generative trajectories
- attractive when the end goal is to generate coherent long-horizon futures rather than only the next event

Why I would not start there:

- for this simulator, a flow-matching system is probably overkill before we establish strong continuous-time forecasting baselines
- evaluation and debugging are harder than with point-process models

### SDE / jump-diffusion temporal point processes

Reference:

- Neural Jump-Diffusion Temporal Point Processes
  - https://proceedings.mlr.press/v235/zhang24cm.html

Why it is promising:

- adds a more expressive continuous-time latent dynamic than simpler TPPs
- looks especially useful when latent system state evolves between observed events

## Practical recommendation

For this repo, the next best ladder is:

1. keep strong mechanistic and empirical baselines
2. add a learned next-event model from the neural TPP family
3. only then test Flow Matching or another joint generative model for longer rollouts

That sequencing gives us:

- a trustworthy baseline floor
- interpretable failure modes
- a comparison surface for future EventFlow-like models

## Current repo status

The repo now has the first learned step in that ladder:

- a GRU-based neural next-event predictor trained on simulated traces
- side-by-side comparison against the empirical and mechanistic baselines
- a browser dashboard for timing and classification comparisons

This is not yet a full neural Hawkes implementation, but it is the right intermediate rung:

- it learns from asynchronous event history
- it predicts both event type and event time
- it gives us a clear benchmark before investing in a heavier continuous-time intensity model
