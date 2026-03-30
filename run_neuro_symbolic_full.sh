#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

exec ./.venv/bin/python compare_models.py \
  --models neuro_symbolic_tpp,multitask_neural_tpp,transformer_tpp,continuous_tpp \
  --train-runs 60 \
  --eval-runs 20 \
  --duration 300 \
  --epochs 14 \
  --control-mode adaptive \
  --simulation-profile richer \
  --output-dir analysis/adaptive_richer_60_20_neuro_symbolic_learned_only \
  --cache-dir analysis/cache_adaptive_richer_60_20_rebalanced \
  --device cpu
