#!/usr/bin/env bash
# L1–L4 KernelBench (original + popcorn): single-turn, default tools, then all tools.
# Uses the same [[models]] block as configs/sweep.l5_hw_translation_all.toml.
#
# Env:
#   STALE_EVALQ_LINES  — passed to run_sweep_eval_stall_watchdog.py (default 5).
#                        Kill when the same `[eval_q]  queue depths: ...` line repeats
#                        more than this many times in a row (~1 line/minute from
#                        run_sweep.py).
#   MAX_RETRIES        — max watchdog restarts per TOML (default 0 = unlimited).
#
# Watchdog exits 125 on stall; only that exit is retried. Other failures abort.
#
#   ./scripts/run_l1234_popcorn_matrix_sweeps.sh
#   STALE_EVALQ_LINES=5 MAX_RETRIES=20 ./scripts/run_l1234_popcorn_matrix_sweeps.sh
#   (set MAX_RETRIES>0 to cap stall retries)

set -euo pipefail
cd "$(dirname "$0")/.."

STALE_EVALQ_LINES="${STALE_EVALQ_LINES:-5}"
MAX_RETRIES="${MAX_RETRIES:-0}"
EXIT_STALE_EVALQ=125

CONFIGS=(
  #configs/sweep.l1234_popcorn_single_turn.toml
  #configs/sweep.l1234_popcorn_default.toml
  configs/sweep.l1234_popcorn_all.toml
)

run_one_cfg() {
  local cfg="$1"
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    echo
    echo "================================================================"
    echo "▶ $(date -u +'%Y-%m-%dT%H:%M:%SZ')  $cfg  (attempt $attempt)"
    echo "================================================================"

    set +e
    uv run python scripts/run_sweep_eval_stall_watchdog.py \
      --stale-run "$STALE_EVALQ_LINES" \
      "$cfg"
    local ec=$?
    set -e

    if (( ec == 0 )); then
      echo "✓ completed $cfg"
      return 0
    fi

    if (( ec == EXIT_STALE_EVALQ )); then
      echo "[run_l1234] watchdog: stale eval_q depth snapshots; will rerun $cfg" >&2
      if (( MAX_RETRIES > 0 && attempt >= MAX_RETRIES )); then
        echo "[run_l1234] FATAL: exceeded MAX_RETRIES=$MAX_RETRIES for $cfg" >&2
        exit "$EXIT_STALE_EVALQ"
      fi
      sleep 2
      continue
    fi

    echo "[run_l1234] FATAL: $cfg exited with $ec (not a watchdog stall)" >&2
    exit "$ec"
  done
}

for cfg in "${CONFIGS[@]}"; do
  run_one_cfg "$cfg"
done

echo "l1234 popcorn matrix sweeps complete at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
