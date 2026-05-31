#!/usr/bin/env bash
# Run eval-only popcorn stress re-eval for the three L1–L3 popcorn GPT sweeps.
# Each destination run name ends with "_gpt" (…_stress_<timestamp>_gpt) for downstream recognition.
#
# Prerequisites:
#   - Completed source runs under runs/ (from sweep.l1234_popcorn_*.toml).
#   - Stress refs generated (use the same interpreter you use here), e.g.:
#       ./scripts/run_popcorn_l123_stress_reval.sh   # prints which python it picked
#       <that-python> scripts/gen_popcorn_stress_refs.py
#
# Usage:
#   ./scripts/run_popcorn_l123_stress_reval.sh
#   STRESS_REFS=KernelBench/stress_refs ./scripts/run_popcorn_l123_stress_reval.sh
#   NUM_GPUS=4 ./scripts/run_popcorn_l123_stress_reval.sh   # parallel workers (default 8)
#   ./scripts/run_popcorn_l123_stress_reval.sh --build-report
#
# Interpreter (first match wins):
#   KERNELBENCH_PYTHON — explicit python binary
#   else repo .venv/bin/python if executable
#   else "uv run python" if uv is on PATH
#   else python3
#
# Scratch paths (temp + caches; default /scratch/abaner — see kernelbench_use_scratch.sh):
#   KERNELBENCH_SCRATCH=/scratch/you ./scripts/run_popcorn_l123_stress_reval.sh
#
# Optional libstdc++ for CUDA extension CXXABI (prepend to LD_PRELOAD):
#   KERNELBENCH_LIBSTDCXX=/path/to/libstdc++.so.6 ./scripts/run_popcorn_l123_stress_reval.sh
# You can also export LD_PRELOAD yourself before invoking this script.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Temp, pip/uv/torch/triton/cuda caches -> /scratch (root FS is often full).
# shellcheck source=scripts/kernelbench_use_scratch.sh
source "$ROOT/scripts/kernelbench_use_scratch.sh"
echo "[run_popcorn_l123_stress_reval] KERNELBENCH_SCRATCH=$KERNELBENCH_SCRATCH TMPDIR=$TMPDIR" >&2

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

STRESS_REFS="${STRESS_REFS:-KernelBench/stress_refs}"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-report) EXTRA+=(--build-report) ; shift ;;
    -h|--help)
      echo "Usage: $0 [--build-report]"
      echo "  Env: STRESS_REFS (default: KernelBench/stress_refs)"
      echo "  Env: NUM_GPUS (default: 8) — passed to reval_popcorn_stress_sweep.py --num-gpus"
      echo "  Env: KERNELBENCH_PYTHON — python to use (overrides .venv / uv / python3)"
      echo "  Env: KERNELBENCH_LIBSTDCXX — path to libstdc++.so.6; prepended to LD_PRELOAD if set"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -n "${KERNELBENCH_PYTHON:-}" ]]; then
  KB_PY_DESC="$KERNELBENCH_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  KB_PY_DESC="$ROOT/.venv/bin/python (repo .venv)"
elif command -v uv >/dev/null 2>&1; then
  KB_PY_DESC="uv run python (project)"
else
  KB_PY_DESC="$(command -v python3)"
fi
echo "[run_popcorn_l123_stress_reval] python: $KB_PY_DESC" >&2
if [[ -n "${LD_PRELOAD:-}" ]]; then
  echo "[run_popcorn_l123_stress_reval] LD_PRELOAD=$LD_PRELOAD" >&2
fi

if [[ ! -d "$ROOT/$STRESS_REFS/large/level1/popcorn" ]]; then
  echo "ERROR: stress refs missing under $ROOT/$STRESS_REFS (run gen_popcorn_stress_refs.py first)." >&2
  exit 1
fi

# Source run directory names (must match runs/<name>/ after a sweep).
# Destination names: <src>_stress_<timestamp>_gpt (always ends with "_gpt").
declare -a SRC_NAMES=(
  "pop_l123_all_gpt"
  "pop_l123_default_gpt"
  "pop_l123_st_gpt"
)

stress_dst_run_name() {
  local name="$1"
  printf '%s_stress_%s_gpt' "$name" "$(date +%Y%m%d_%H%M%S_%N)"
}

for name in "${SRC_NAMES[@]}"; do
  src="runs/$name"
  if [[ ! -d "$src" ]]; then
    echo "ERROR: missing source run: $ROOT/$src" >&2
    exit 1
  fi
done

for name in "${SRC_NAMES[@]}"; do
  dst="$(stress_dst_run_name "$name")"
  echo "========== stress reval: runs/$name  ->  runs/$dst =========="
  kb_python scripts/reval_popcorn_stress_sweep.py \
    --src-run "runs/$name" \
    --dst-run-name "$dst" \
    --dst-exact-name \
    --stress-refs "$STRESS_REFS" \
    --num-gpus "${NUM_GPUS:-8}" \
    --num-correct-trials 5 \
    --num-perf-trials 5 \
    "${EXTRA[@]}"
done

echo "Done. Stress runs (names end with _gpt): ls -dt runs/pop_l123_*_stress_*_gpt | head"
