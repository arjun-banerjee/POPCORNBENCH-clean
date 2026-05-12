#!/usr/bin/env bash
# Run all Level-5 hardware-translation 8xH100 sweep variants (single-turn,
# default tools, all tools).
#
# Mirrors scripts/run_l5_hw_translation_sweeps.sh: baseline first, then richer
# tool sets. Unlike that 1xH100 driver, every variant here runs ONE work item
# at a time across all 8 GPUs — submit_kernel / run_correctness / profile_kernel
# each spawn an 8-rank torchrun + NCCL subprocess. Wall-clock is
# N_problems * N_models * (LLM_time + eval_time); plan on multiple hours per
# (problem, model) pair, and many hours per (config, model).
#
# Models are listed in each configs/sweep.l5_hw_translation_8xh100*.toml;
# include GPT-5.5 (POPCORN_CENTRAL_AZURE_KEY), FW-GLM-5-1 (POPCORN_AZURE_KEY),
# Llama-Maverick (THAVA_AZURE_KEY), Grok (XAI_API_KEY).
#
# Sanity checks before launch:
#   - nvidia-smi reports 8 GPUs
#   - $POPCORN_CENTRAL_AZURE_KEY, $POPCORN_AZURE_KEY, $THAVA_AZURE_KEY,
#     $XAI_API_KEY are all set
#   - `which ncu` returns a path (profile_kernel in the `_all` variant
#     hard-fails otherwise)
#
# Usage:
#   ./scripts/run_l5_hw_translation_8xh100_sweeps.sh
#   ./scripts/run_l5_hw_translation_8xh100_sweeps.sh 2>&1 | tee \
#       l5_hw_translation_8xh100_sweeps.log

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
  configs/sweep.l5_hw_translation_8xh100_single_turn.toml
  configs/sweep.l5_hw_translation_8xh100_default.toml
  configs/sweep.l5_hw_translation_8xh100_all.toml
)

run_one() {
  local cfg="$1"
  echo
  echo "================================================================"
  echo "▶ $(date -u +%Y-%m-%dT%H:%M:%SZ)  $cfg"
  echo "================================================================"
  uv run python scripts/run_sweep.py "$cfg"
}

for c in "${CONFIGS[@]}"; do run_one "$c"; done

echo "l5 8xh100 hardware translation sweeps complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
