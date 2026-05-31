#!/usr/bin/env bash
# L1–L3 popcorn2: single-turn, default tools, then all tools (names end with _gpt).
# submit_kernel uses stress_refs2 (large/awkward/xl) × num_correct_trials/tier.
# run_correctness (default/all) uses canonical popcorn2 refs only.
#
# Prerequisites:
#   uv run python scripts/gen_popcorn2_stress_refs.py
#
# Env:
#   STALE_EVALQ_LINES  — watchdog: stale eval_q depth lines (default 5).
#   MAX_RETRIES        — max watchdog restarts per TOML (0 = unlimited).
#   KERNELBENCH_SCRATCH — see scripts/kernelbench_use_scratch.sh
#   KERNELBENCH_PYTHON — explicit python (else .venv / uv / python3)
#   KERNELBENCH_LIBSTDCXX — libstdc++.so.6 for LD_PRELOAD (CUDA extension CXXABI)
#
# Watchdog exits 125 on stall; retries with exponential backoff (cap 60s).
#
#   ./scripts/run_l1234_popcorn2_matrix_sweeps.sh
#   STALE_EVALQ_LINES=5 MAX_RETRIES=20 ./scripts/run_l1234_popcorn2_matrix_sweeps.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Temp, pip/uv/torch/triton/cuda caches -> /scratch (root FS is often full).
# shellcheck source=scripts/kernelbench_use_scratch.sh
source "$ROOT/scripts/kernelbench_use_scratch.sh"
echo "[run_l1234_popcorn2] KERNELBENCH_SCRATCH=$KERNELBENCH_SCRATCH TMPDIR=$TMPDIR" >&2

export LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"

if [[ -n "${KERNELBENCH_LIBSTDCXX:-}" ]]; then
  if [[ ! -f "$KERNELBENCH_LIBSTDCXX" ]]; then
    echo "ERROR: KERNELBENCH_LIBSTDCXX is not a file: $KERNELBENCH_LIBSTDCXX" >&2
    exit 1
  fi
  case ":${LD_PRELOAD:-}:" in
    *:"$KERNELBENCH_LIBSTDCXX":*) ;;
    *) export LD_PRELOAD="${KERNELBENCH_LIBSTDCXX}${LD_PRELOAD:+:$LD_PRELOAD}" ;;
  esac
fi

kb_python() {
  if [[ -n "${KERNELBENCH_PYTHON:-}" ]]; then
    if [[ ! -x "$KERNELBENCH_PYTHON" ]]; then
      echo "ERROR: KERNELBENCH_PYTHON is not executable: $KERNELBENCH_PYTHON" >&2
      exit 1
    fi
    "$KERNELBENCH_PYTHON" "$@"
  elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    "$ROOT/.venv/bin/python" "$@"
  elif command -v uv >/dev/null 2>&1; then
    uv run python "$@"
  else
    if ! command -v python3 >/dev/null 2>&1; then
      echo "ERROR: no python3 on PATH and no .venv or uv" >&2
      exit 1
    fi
    python3 "$@"
  fi
}

STRESS_REFS2="${STRESS_REFS2:-KernelBench/stress_refs2}"
if [[ ! -d "$ROOT/$STRESS_REFS2/large/level1/popcorn2" ]]; then
  echo "ERROR: missing $ROOT/$STRESS_REFS2 (run: kb_python scripts/gen_popcorn2_stress_refs.py)" >&2
  exit 1
fi

STALE_EVALQ_LINES="${STALE_EVALQ_LINES:-5}"
MAX_RETRIES="${MAX_RETRIES:-0}"
EXIT_STALE_EVALQ=125

CONFIGS=(
  configs/sweep.l1234_popcorn2_single_turn.toml
  configs/sweep.l1234_popcorn2_default.toml
  configs/sweep.l1234_popcorn2_all.toml
)

run_one_cfg() {
  local cfg="$1"
  local attempt=0
  local delay_s=2.0
  while true; do
    attempt=$((attempt + 1))
    echo
    echo "================================================================"
    echo "▶ $(date -u +'%Y-%m-%dT%H:%M:%SZ')  $cfg  (attempt $attempt)"
    echo "================================================================"

    set +e
    kb_python scripts/run_sweep_eval_stall_watchdog.py \
      --stale-run "$STALE_EVALQ_LINES" \
      "$cfg"
    local ec=$?
    set -e

    if (( ec == 0 )); then
      echo "✓ completed $cfg"
      return 0
    fi

    if (( ec == EXIT_STALE_EVALQ )); then
      echo "[run_l1234_popcorn2] watchdog: stale eval_q; retry $cfg in ${delay_s}s" >&2
      if (( MAX_RETRIES > 0 && attempt >= MAX_RETRIES )); then
        echo "[run_l1234_popcorn2] FATAL: exceeded MAX_RETRIES=$MAX_RETRIES for $cfg" >&2
        exit "$EXIT_STALE_EVALQ"
      fi
      sleep "$(printf '%.0f' "$delay_s")"
      delay_s="$(python3 -c "print(min(${delay_s} * 1.5, 60.0))")"
      continue
    fi

    echo "[run_l1234_popcorn2] FATAL: $cfg exited with $ec (not a watchdog stall)" >&2
    exit "$ec"
  done
}

for cfg in "${CONFIGS[@]}"; do
  run_one_cfg "$cfg"
done

echo "l123 popcorn2 matrix sweeps complete at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "Runs: ls -dt runs/pop_l123_popcorn2_*_gpt | head"
