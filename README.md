# Traffic Intersection Control Simulation

This repository now contains a first-pass discrete-event simulation for a four-way signalized intersection.

The simulator supports two signal-control modes:

- `adaptive` (default): green durations respond to queue pressure and arrival-rate pressure
- `fixed`: restores the original fixed-time controller

It also supports two traffic-dynamics profiles:

- `richer` (default): bursty time-varying arrivals plus lane-specific service headways
- `baseline`: the earlier stationary-arrival, shared-headway setup

## What You Need To Run

If you just want the main current workflow, use this checklist.

### 1. Local setup

Create and activate the virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Run the simulator locally

Use the current default setup:

```bash
python3 simulation.py --duration 300 --seed 7 --control-mode adaptive --simulation-profile richer
```

### 3. Run a local baseline benchmark

This is the cheapest way to regenerate a comparison report locally:

```bash
./.venv/bin/python compare_models.py \
  --control-mode adaptive \
  --simulation-profile richer \
  --train-runs 60 \
  --eval-runs 20 \
  --duration 300 \
  --epochs 14 \
  --cache-dir analysis/cache_adaptive_richer_60_20_rebalanced \
  --output-dir analysis/adaptive_richer_60_20_rebalanced_full
```

Open the results:

```bash
open analysis/adaptive_richer_60_20_rebalanced_full/model_comparison.html
open analysis/adaptive_richer_60_20_rebalanced_full/traffic_predictions.html
```

### 4. Run learned-model training on Colab GPU

Open `notebooks/colab_gpu_benchmark.ipynb` in Colab, switch the runtime to `GPU`, then run the cells top to bottom.

The main training command in the notebook is:

```bash
python train_learned.py \
  --device cuda \
  --control-mode adaptive \
  --simulation-profile richer \
  --train-runs 60 \
  --eval-runs 20 \
  --duration 300 \
  --epochs 14 \
  --cache-dir analysis/cache_adaptive_richer_60_20_rebalanced \
  --output-dir analysis/adaptive_richer_60_20_rebalanced_learned \
  --checkpoint-dir analysis/adaptive_richer_60_20_rebalanced_learned/checkpoints
```

### 5. What to download back from Colab

Bring back either the full artifact zip or at least these files:

- `analysis/adaptive_richer_60_20_rebalanced_learned/model_comparison.json`
- `analysis/adaptive_richer_60_20_rebalanced_learned/model_comparison.html`
- `analysis/adaptive_richer_60_20_rebalanced_learned/runtime_summary.json`
- `analysis/adaptive_richer_60_20_rebalanced_learned/checkpoints/*.pt`

### 6. Compare Colab runtime to local

After you have both runtime summaries:

```bash
./.venv/bin/python compare_runtime.py \
  analysis/runtime_local/runtime_summary.json \
  analysis/adaptive_richer_60_20_rebalanced_learned/runtime_summary.json
```

## What it models

- Four inbound lanes: `north`, `south`, `east`, `west`
- A signal controller with phases:
  - `NS_GREEN`
  - `ALL_RED`
  - `EW_GREEN`
  - `ALL_RED`
- Stochastic vehicle arrivals using exponential interarrival times
- In the richer profile, time-varying corridor demand pulses and bursty platoon arrivals
- Vehicle queueing and service; in the richer profile, service rates vary by lane

## Run it

```bash
python3 simulation.py --duration 300 --seed 7
```

To run the original fixed-time controller instead:

```bash
python3 simulation.py --duration 300 --seed 7 --control-mode fixed
```

To reproduce the earlier simpler environment instead of the richer default:

```bash
python3 simulation.py --duration 300 --seed 7 --simulation-profile baseline
```

Outputs are written to `output/`:

- `summary.json`: aggregate lane metrics
- `events.csv`: event-by-event trace for later EventFlow-style modeling
- `index.html`: self-contained browser replay of the scenario outcome

Open the animation directly in a browser:

```bash
open output/index.html
```

## Compare Forecasting Baselines

Generate training and evaluation runs, score several next-event predictors, and emit a browser dashboard:

```bash
./.venv/bin/python compare_models.py --train-runs 24 --eval-runs 6 --duration 300 --epochs 14
open analysis/model_comparison.html
```

For faster iteration while tuning a single model, reuse cached simulator runs and restrict the benchmark:

```bash
./.venv/bin/python compare_models.py \
  --train-runs 24 \
  --eval-runs 6 \
  --duration 300 \
  --epochs 18 \
  --models transformer_tpp \
  --cache-dir analysis/cache
```

Artifacts are written to `analysis/`:

- `model_comparison.json`: raw metrics and example predictions
- `model_comparison.html`: browser dashboard for prediction-vs-actual comparison

The comparison currently includes:

- simple empirical baselines
- a mechanistic phase-aware baseline
- a learned GRU-based neural temporal event predictor
- a multitask GRU baseline with direct fixed-horizon condition heads
- a continuous-time LSTM-style temporal point process baseline
- an attention-based temporal point process baseline

The dashboard now includes both:

- short-horizon next-event metrics
- longer-horizon rollout metrics for event prediction

Useful CLI options:

- `--models`: run only a comma-separated subset such as `transformer_tpp` or `neural_tpp,continuous_tpp`
- `--cache-dir`: cache generated simulation runs so repeated comparisons do not regenerate the same data
- `--device`: choose `auto`, `cpu`, `cuda`, or `mps` for the learned models
- `--control-mode`: choose `adaptive` or `fixed` when generating comparison runs
- `--simulation-profile`: choose `richer` or `baseline` when generating comparison runs

Project planning and modeling follow-ups are tracked in `SUGGESTION_TRACKER.md`.

See `MODELING_APPROACHES.md` for the short-list of modeling families to try next.

## Colab GPU Workflow

The learned models can now train on GPU through the same codepath used locally. In Colab, after cloning or uploading the repo:

```bash
pip install -e .[colab]
python train_learned.py \
  --device cuda \
  --control-mode adaptive \
  --simulation-profile richer \
  --train-runs 60 \
  --eval-runs 20 \
  --duration 300 \
  --epochs 14 \
  --cache-dir analysis/cache_adaptive_richer_60_20_rebalanced \
  --output-dir analysis/adaptive_richer_60_20_rebalanced_learned \
  --checkpoint-dir analysis/adaptive_richer_60_20_rebalanced_learned/checkpoints
```

That writes a learned-model-only comparison plus saved checkpoints to:

- `analysis/adaptive_richer_60_20_rebalanced_learned/model_comparison.json`
- `analysis/adaptive_richer_60_20_rebalanced_learned/model_comparison.html`
- `analysis/adaptive_richer_60_20_rebalanced_learned/runtime_summary.json`
- `analysis/adaptive_richer_60_20_rebalanced_learned/checkpoints/*.pt`

If you want the full benchmark on GPU for the learned models while keeping the hand-built baselines in the mix:

```bash
python compare_models.py \
  --device cuda \
  --control-mode adaptive \
  --simulation-profile richer \
  --train-runs 60 \
  --eval-runs 20 \
  --duration 300 \
  --epochs 14 \
  --cache-dir analysis/cache_adaptive_richer_60_20_rebalanced \
  --output-dir analysis/adaptive_richer_60_20_rebalanced_full
```

There is also a Colab starter notebook at `notebooks/colab_gpu_benchmark.ipynb` with:

- setup cells
- the updated richer-simulator learned-model training command
- runtime summary preview
- artifact zip/download cells for bringing the benchmark outputs and checkpoints back from Colab

## Why this is a good EventFlow precursor

The trace already exposes the core event types we will want to map into an EventFlow-like structure:

- `phase_change`
- `vehicle_arrival`
- `vehicle_departure`

From here, we can promote:

- lane queues into entity state
- signal phases into controller state
- event callbacks into explicit event nodes and transitions
