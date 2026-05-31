#!/usr/bin/env bash
# popcorn2 subbench (12+12+10 slate): resume agent sweeps into bench run dirs.
#
# Slate: configs/popcorn2_bench_slate_v1.toml
#   L1×12 + L2×12 + L3×10 = 34 problems × 3 models = 102 cells per tier.
#
# Typical workflow:
#   1. Export any finished cells from full sweeps (no GPU):
#        EXPORT_FIRST=1 ./scripts/run_popcorn2_bench.sh
#   2. Fill gaps (all tier):
#        ./scripts/run_popcorn2_bench.sh
#   3. Or explicitly:
#        TIER=all ./scripts/run_popcorn2_bench.sh
#
# Output:
#   runs/pop_l123_popcorn2_bench_all_gpt/
#
# Env:
#   TIER              — all  (default: all)
#   EXPORT_FIRST      — 1 = run export_popcorn2_bench_slate.py before sweep
#   BUILD_REPORT      — 1 = build_report.py after each tier (default: 1)
#   STALE_EVALQ_LINES — watchdog stale eval_q lines (default 5)
#   MAX_RETRIES       — max watchdog restarts per TOML (0 = unlimited)
#   KERNELBENCH_SCRATCH, KERNELBENCH_PYTHON, KERNELBENCH_LIBSTDCXX — see below
#
# Prerequisites:
#   uv run python scripts/gen_popcorn2_stress_refs.py

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=scripts/kernelbench_use_scratch.sh
source "$ROOT/scripts/kernelbench_use_scratch.sh"
echo "[run_popcorn2_bench] KERNELBENCH_SCRATCH=$KERNELBENCH_SCRATCH TMPDIR=$TMPDIR" >&2

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

TIER="${TIER:-all}"
EXPORT_FIRST="${EXPORT_FIRST:-0}"
BUILD_REPORT="${BUILD_REPORT:-1}"
STALE_EVALQ_LINES="${STALE_EVALQ_LINES:-5}"
MAX_RETRIES="${MAX_RETRIES:-0}"
EXIT_STALE_EVALQ=125

case "$TIER" in
  all)
    CONFIGS=(configs/sweep.l1234_popcorn2_bench_all.toml)
    EXPORT_TIERS=(all)
    RUN_DIRS=(pop_l123_popcorn2_bench_all_gpt)
    ;;
  # default)
  #   CONFIGS=(configs/sweep.l1234_popcorn2_bench_default.toml)
  #   EXPORT_TIERS=(default)
  #   RUN_DIRS=(pop_l123_popcorn2_bench_default_gpt)
  #   ;;
  # st)
  #   CONFIGS=(configs/sweep.l1234_popcorn2_bench_st.toml)
  #   EXPORT_TIERS=(st)
  #   RUN_DIRS=(pop_l123_popcorn2_bench_st_gpt)
  #   ;;
  # both)
  #   CONFIGS=(
  #     configs/sweep.l1234_popcorn2_bench_st.toml
  #     configs/sweep.l1234_popcorn2_bench_default.toml
  #   )
  #   EXPORT_TIERS=(st default)
  #   RUN_DIRS=(
  #     pop_l123_popcorn2_bench_st_gpt
  #     pop_l123_popcorn2_bench_default_gpt
  #   )
  #   ;;
  *)
    echo "ERROR: TIER must be all (got: $TIER)" >&2
    exit 1
    ;;
esac

if (( EXPORT_FIRST )); then
  for et in "${EXPORT_TIERS[@]}"; do
    echo "[run_popcorn2_bench] export --tier $et --force"
    kb_python scripts/export_popcorn2_bench_slate.py --tier "$et" --force
  done
fi

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
      echo "[run_popcorn2_bench] watchdog: stale eval_q; retry $cfg in ${delay_s}s" >&2
      if (( MAX_RETRIES > 0 && attempt >= MAX_RETRIES )); then
        echo "[run_popcorn2_bench] FATAL: exceeded MAX_RETRIES=$MAX_RETRIES for $cfg" >&2
        exit "$EXIT_STALE_EVALQ"
      fi
      sleep "$(printf '%.0f' "$delay_s")"
      delay_s="$(python3 -c "print(min(${delay_s} * 1.5, 60.0))")"
      continue
    fi

    echo "[run_popcorn2_bench] FATAL: $cfg exited with $ec (not a watchdog stall)" >&2
    exit "$ec"
  done
}

for cfg in "${CONFIGS[@]}"; do
  run_one_cfg "$cfg"
done

if (( BUILD_REPORT )); then
  for run_dir in "${RUN_DIRS[@]}"; do
    if [[ -d "$ROOT/runs/$run_dir" ]]; then
      echo "[run_popcorn2_bench] build_report runs/$run_dir"
      kb_python scripts/build_report.py "runs/$run_dir"
    fi
  done
fi

echo "popcorn2 bench complete at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "  All: runs/pop_l123_popcorn2_bench_all_gpt/report/index.html"
# echo "  ST:      runs/pop_l123_popcorn2_bench_st_gpt/report/index.html"
# echo "  Default: runs/pop_l123_popcorn2_bench_default_gpt/report/index.html"
