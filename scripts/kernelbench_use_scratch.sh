#!/usr/bin/env bash
# Route temp files and tool caches to /scratch (avoid filling $HOME on root FS).
# Source from other scripts:
#   source "$(dirname "${BASH_SOURCE[0]}")/kernelbench_use_scratch.sh"
#
# Override root:  KERNELBENCH_SCRATCH=/other/scratch/path

: "${KERNELBENCH_SCRATCH:=/scratch/abaner}"

_kb_scratch_dirs=(
  "${KERNELBENCH_SCRATCH}/tmp"
  "${KERNELBENCH_SCRATCH}/.cache"
  "${KERNELBENCH_SCRATCH}/.local/share"
  "${KERNELBENCH_SCRATCH}/torch_extensions"
  "${KERNELBENCH_SCRATCH}/triton"
  "${KERNELBENCH_SCRATCH}/cuda_cache"
)
for _d in "${_kb_scratch_dirs[@]}"; do
  mkdir -p "$_d"
done

export TMPDIR="${KERNELBENCH_SCRATCH}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export XDG_CACHE_HOME="${KERNELBENCH_SCRATCH}/.cache"
export XDG_DATA_HOME="${KERNELBENCH_SCRATCH}/.local/share"
export UV_CACHE_DIR="${KERNELBENCH_SCRATCH}/.cache/uv"
export PIP_CACHE_DIR="${KERNELBENCH_SCRATCH}/.cache/pip"
export HF_HOME="${HF_HOME:-${KERNELBENCH_SCRATCH}/.cache/huggingface}"
export TORCH_HOME="${KERNELBENCH_SCRATCH}/.cache/torch"
export TORCH_EXTENSIONS_DIR="${KERNELBENCH_SCRATCH}/torch_extensions"
export TRITON_CACHE_DIR="${KERNELBENCH_SCRATCH}/triton"
export CUDA_CACHE_PATH="${KERNELBENCH_SCRATCH}/cuda_cache"
export MPLCONFIGDIR="${KERNELBENCH_SCRATCH}/.cache/matplotlib"

mkdir -p \
  "$UV_CACHE_DIR" \
  "$PIP_CACHE_DIR" \
  "$HF_HOME" \
  "$TORCH_HOME" \
  "$MPLCONFIGDIR"
