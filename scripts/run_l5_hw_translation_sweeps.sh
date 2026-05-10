#!/usr/bin/env bash
# Run all Level-5 hardware-translation sweep variants (single-turn, default tools, all tools).
#
# Order matches scripts/run_demo_sweeps.sh: baseline first, then richer tool sets.
#
#   ./scripts/run_l5_hw_translation_sweeps.sh
#   ./scripts/run_l5_hw_translation_sweeps.sh 2>&1 | tee l5_hw_translation_sweeps.log

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
  configs/sweep.l5_hw_translation_single_turn.toml
  configs/sweep.l5_hw_translation_default.toml
  configs/sweep.l5_hw_translation_all.toml
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

echo "l5 hardware translation sweeps complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
