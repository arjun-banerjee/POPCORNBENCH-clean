"""Reference-runtime probe used to derive dynamic eval timeouts.

Background
----------
The static ``eval_torchrun_timeout_s`` from a sweep config (e.g. 5400s for
the 8xH100 default) is a *hard ceiling* for the torchrun subprocess. For
hard problems that's appropriate, but it also means a hung or pathological
candidate kernel can burn ~30 minutes per call before being killed.

This module probes the reference (PyTorch ``Model``) forward time once per
(problem, world_size) for the current sweep worker. Callers then derive
per-tool dynamic timeouts via simple ``overhead + k * n_trials * t_ref``
formulas, clipped to a configured floor / ceiling.

Design choices
--------------
1. **Multi-rank path mirrors ``eval_kernel_via_torchrun``.** When
   ``world_size > 1`` we spawn a torchrun subprocess that runs the reference
   forward only (no custom kernel involved), so the probe wall time matches
   the operational cost of correctness / submit_kernel evals on the same
   topology. Reuses the existing ``--worker`` entrypoint in
   ``distributed_torchrun_eval`` via a new ``mode="ref_only"`` request flag.

2. **Single-GPU path is in-process.** Uses the existing
   ``measure_ref_program_time`` helper from ``kernelbench.timing`` for a
   pure PyTorch reference. CUDA events with N trials + small warmup; this
   does not touch any candidate kernel state.

3. **Always fail-soft.** Any error (compile failure, CUDA error, torchrun
   exit code != 0, parse failure, timeout) returns ``None`` so callers fall
   back to ``eval_torchrun_timeout_s`` and continue serving requests.

4. **Bounded by its own ``probe_timeout_s``.** If the reference itself is
   slow enough that the probe times out, returning ``None`` tells callers
   to fall back to the ceiling — the right thing to do when the problem is
   too large for any sub-ceiling cap to be safe.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# Public API ----------------------------------------------------------------


def probe_reference_runtime(
    *,
    ref_arch_src: str,
    world_size: int,
    num_trials: int,
    backend: str,
    precision_str: str,
    device: Optional[torch.device] = None,
    probe_timeout_s: int = 300,
    build_dir: Optional[str] = None,
    verbose: bool = False,
    seed_num: int = 42,
) -> Optional[float]:
    """Return mean reference-forward seconds, or ``None`` on any failure.

    Parameters
    ----------
    ref_arch_src
        Reference PyTorch source (the ``Model`` definition file).
    world_size
        Multi-rank world size used by the actual eval. When >1 the probe is
        run under a torchrun subprocess so the cost includes init_process_group
        + NCCL barriers (matching what ``run_correctness`` / ``submit_kernel``
        pay).
    num_trials
        Number of timed reference forwards to average. A small number (e.g.
        5) is sufficient — we just need an order-of-magnitude estimate.
    backend, precision_str
        Forwarded to the underlying timer so dtype/device behaviour matches
        the candidate eval.
    device
        Used only by the single-GPU path. Multi-rank picks per-rank devices
        from ``LOCAL_RANK`` inside the torchrun worker.
    probe_timeout_s
        Hard ceiling for the probe itself (subprocess timeout for the
        multi-rank path; soft for the single-GPU path via Python-level
        time check after each trial).
    """

    try:
        if world_size and world_size > 1:
            return _probe_via_torchrun(
                ref_arch_src=ref_arch_src,
                world_size=int(world_size),
                num_trials=int(num_trials),
                backend=backend,
                precision_str=precision_str,
                probe_timeout_s=int(probe_timeout_s),
                build_dir=build_dir,
                verbose=verbose,
                seed_num=seed_num,
            )
        return _probe_in_process(
            ref_arch_src=ref_arch_src,
            num_trials=int(num_trials),
            backend=backend,
            precision_str=precision_str,
            device=device,
            probe_timeout_s=int(probe_timeout_s),
            verbose=verbose,
            seed_num=seed_num,
        )
    except Exception as e:
        logger.warning(
            "[reference_timing] probe failed (world=%s): %s: %s",
            world_size, type(e).__name__, e,
        )
        return None


# Timeout helpers -----------------------------------------------------------


def compute_dynamic_timeout(
    *,
    t_ref_s: Optional[float],
    overhead_s: float,
    k: float,
    n_trials: int,
    floor_s: float,
    ceiling_s: float,
) -> int:
    """Apply ``min(ceiling, max(floor, overhead + k * n_trials * t_ref))``.

    Returns ``ceiling_s`` when ``t_ref_s`` is None (probe failure or
    feature disabled). Always returns an int.
    """
    if t_ref_s is None or t_ref_s <= 0:
        return int(ceiling_s)
    raw = overhead_s + k * float(n_trials) * float(t_ref_s)
    return int(min(float(ceiling_s), max(float(floor_s), raw)))


# Single-GPU (in-process) probe --------------------------------------------


def _probe_in_process(
    *,
    ref_arch_src: str,
    num_trials: int,
    backend: str,
    precision_str: str,
    device: Optional[torch.device],
    probe_timeout_s: int,
    verbose: bool,
    seed_num: int,
) -> Optional[float]:
    """Time the reference forward in this process; return mean seconds."""
    if not torch.cuda.is_available():
        return None

    if device is None:
        device = torch.device("cuda")

    from kernelbench.eval import (
        load_original_model_and_inputs,
        set_seed,
        get_torch_dtype_from_string,
    )

    context: dict = {}
    loaded = load_original_model_and_inputs(ref_arch_src, context)
    if loaded is None:
        return None
    Model, get_init_inputs, get_inputs = loaded
    if Model is None or get_init_inputs is None or get_inputs is None:
        return None

    precision_dtype = get_torch_dtype_from_string(precision_str)

    def _to_dev(t):
        if not isinstance(t, torch.Tensor):
            return t
        if not t.is_floating_point():
            return t.to(device=device)
        return t.to(device=device, dtype=precision_dtype)

    torch.cuda.set_device(device)
    set_seed(seed_num)
    init_inputs = [_to_dev(x) for x in get_init_inputs()]
    set_seed(seed_num)
    inputs = [_to_dev(x) for x in get_inputs()]

    with torch.no_grad():
        model = Model(*init_inputs).to(device=device, dtype=precision_dtype)

    # Warmup (1 trial) to amortize one-time autotune / cache effects.
    torch.cuda.synchronize(device=device)
    deadline = time.monotonic() + max(1, int(probe_timeout_s))
    try:
        with torch.no_grad():
            _ = model(*inputs)
        torch.cuda.synchronize(device=device)
    except Exception as e:
        logger.warning(
            "[reference_timing] in-process warmup failed: %s: %s",
            type(e).__name__, e,
        )
        return None

    elapsed: list[float] = []
    for _ in range(max(1, num_trials)):
        if time.monotonic() >= deadline:
            if verbose:
                logger.info(
                    "[reference_timing] in-process probe hit timeout after "
                    "%d / %d trials", len(elapsed), num_trials,
                )
            break
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            with torch.no_grad():
                _ = model(*inputs)
        except Exception as e:
            logger.warning(
                "[reference_timing] in-process trial failed: %s: %s",
                type(e).__name__, e,
            )
            return None
        end.record()
        torch.cuda.synchronize(device=device)
        elapsed.append(start.elapsed_time(end) / 1000.0)  # ms -> s

    if not elapsed:
        return None
    return float(sum(elapsed) / len(elapsed))


# Multi-rank (torchrun) probe -----------------------------------------------


def _probe_via_torchrun(
    *,
    ref_arch_src: str,
    world_size: int,
    num_trials: int,
    backend: str,
    precision_str: str,
    probe_timeout_s: int,
    build_dir: Optional[str],
    verbose: bool,
    seed_num: int,
) -> Optional[float]:
    """Spawn torchrun + reference-only worker; return rank-0 mean seconds."""
    from kernelbench.distributed_torchrun_eval import _count_physical_gpus

    if torch.cuda.is_available():
        physical = _count_physical_gpus()
        if physical < world_size:
            if verbose:
                logger.info(
                    "[reference_timing] only %d physical GPU(s) visible, "
                    "want %d; skipping torchrun probe.", physical, world_size,
                )
            return None

    request = {
        "mode": "ref_only",
        "ref_arch_src": ref_arch_src,
        "num_trials": int(num_trials),
        "backend": backend,
        "precision_str": precision_str,
        "seed_num": int(seed_num),
        "verbose": bool(verbose),
        "build_dir": build_dir,
    }

    with tempfile.TemporaryDirectory(prefix="kb_refprobe_") as tmp:
        req_path = os.path.join(tmp, "request.json")
        res_path = os.path.join(tmp, "result.json")
        with open(req_path, "w") as f:
            json.dump(request, f)

        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("HIP_VISIBLE_DEVICES", None)
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env.setdefault("MASTER_PORT", "0")
        env.setdefault("NCCL_DEBUG", "WARN")

        argv = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={world_size}",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=localhost:0",
            "--no-python",
            sys.executable,
            "-m",
            "kernelbench.distributed_torchrun_eval",
            "--worker",
            req_path,
            res_path,
        ]

        try:
            proc = subprocess.run(
                argv,
                env=env,
                capture_output=True,
                text=True,
                timeout=max(30, int(probe_timeout_s)),
            )
        except subprocess.TimeoutExpired:
            if verbose:
                logger.info(
                    "[reference_timing] torchrun probe timed out after %ds",
                    probe_timeout_s,
                )
            return None

        if not os.path.exists(res_path):
            if verbose:
                tail_err = (proc.stderr or "")[-400:]
                logger.info(
                    "[reference_timing] torchrun probe produced no result "
                    "file (rc=%d). stderr tail: %s", proc.returncode, tail_err,
                )
            return None

        try:
            with open(res_path) as f:
                payload = json.load(f)
        except Exception as e:
            if verbose:
                logger.info(
                    "[reference_timing] failed to parse probe result: %s", e,
                )
            return None

        if "error" in payload:
            if verbose:
                logger.info(
                    "[reference_timing] torchrun probe error: %s",
                    payload["error"],
                )
            return None

        try:
            mean_s = float(payload["mean_s"])
            if mean_s <= 0:
                return None
            return mean_s
        except (KeyError, TypeError, ValueError):
            return None
