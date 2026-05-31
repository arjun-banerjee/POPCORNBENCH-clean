#!/usr/bin/env python3
"""
Copy a completed **popcorn** sweep run (e.g. ``pop_l123_all_gpt``), then for each
saved ``level_*_problem_*_kernel.py`` run **eval-only stress**: three reference
trees (``large/``, ``awkward/``, ``xl/``) × ``num_correct_trials`` (default 5)
correctness executions per tier → **15 total** correctness trials per kernel
when all tiers run.

Writes fresh ``final_result`` + ``outcome`` into the **copied** trajectories only
(original ``runs/<src>`` is never modified).

Immediately after the copy, every trajectory gets ``run_name`` set to the
destination run; any ``outcome`` of ``correct`` is cleared to empty so reports
do not show stale green until stress finishes. When a kernel job starts, that
trajectory is marked ``outcome`` = ``in_progress`` until the merged stress
result is written.

Reference layout (``--stress-refs ROOT``)::

    ROOT/large/level{1,2,3}/<variant>/<problem.name>
    ROOT/awkward/level{1,2,3}/<variant>/<problem.name>
    ROOT/xl/level{1,2,3}/<variant>/<problem.name>

``<variant>`` and ``<problem.name>`` must match the source sweep's dataset
(typically ``variants = ["popcorn"]``). Filenames must match the canonical
``KernelBench/level{N}/<variant>/`` problem names (same leading problem id).

Example::

    uv run python scripts/reval_popcorn_stress_sweep.py \\
      --src-run runs/pop_l123_default_gpt \\
      --dst-run-name pop_l123_default_gpt_stress \\
      --stress-refs KernelBench/stress_refs

``--dst-run-name`` is a **base** name: a timestamp suffix is appended by default
(``pop_l123_default_gpt_stress_20260514_153045_123456``) so a new run never
collides with a leftover partial copy. Pass ``--dst-exact-name`` to use the
argument verbatim as the ``runs/<name>`` directory (must not exist).

By default uses **one worker process per GPU** (``--num-gpus``, capped by
``torch.cuda.device_count()``), round-robin assignment of kernels across GPUs
— similar spirit to multi-worker sweep parallelism. Use ``--serial`` for a
single-GPU loop on ``cuda:0``.

Each kernel job runs in a **fresh child process** so an illegal memory access
(or other CUDA context-fatal error) in one candidate cannot poison later
kernels on the same GPU. Within a job, remaining stress tiers are skipped after
the first context-fatal failure on that process.

The script exits with status **1** if any GPU worker exits non-zero or any
kernel job raises before completing (after writing a crash-recovery trajectory
when possible).

L2 multi-GPU popcorn comm kernels are skipped by default (same ids as
``run_sweep.MULTI_GPU_PROBLEMS``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO_TOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_TOP not in sys.path:
    sys.path.insert(0, REPO_TOP)

# Intentionally avoid importing kernelbench.eval / .utils / .agent.tools at module
# load: they pull in torch before multiprocessing spawn children can pin
# CUDA_VISIBLE_DEVICES. Import those inside main() / worker / per-kernel helpers.

from kernelbench.agent.eval_server import _is_cuda_context_fatal
from kernelbench.dataset import construct_kernelbench_dataset

_KERNEL_PATH_RE = re.compile(r"level_(\d+)_problem_(\d+)_kernel\.py$")

# Match scripts/run_sweep.py — L2 popcorn kernels that need multi-GPU eval.
L2_POPCORN_MULTI_GPU_IDS = frozenset({2, 11, 18, 27, 34, 38})


def _resolve_under_repo(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(REPO_TOP, p)


def _force_backend_precision(backend: str, precision: str) -> str:
    b = backend.lower()
    if b == "tilelang":
        return "fp16"
    if b == "thunderkittens":
        return "bf16"
    return precision


def _outcome_from_merged(fr: dict[str, Any] | None) -> str:
    if fr is None:
        return "error"
    if not fr.get("compiled"):
        return "compile_fail"
    if not fr.get("correctness"):
        return "incorrect"
    return "correct"


def _write_trajectory_json(traj_path: Path, traj: dict[str, Any]) -> None:
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(traj, f, indent=2)
        f.write("\n")


def _json_safe_eval_tree(obj: Any) -> Any:
    """Coerce eval result trees to JSON-serializable values (e.g. Exception → str)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, BaseException):
        return f"{type(obj).__name__}: {obj}"
    if isinstance(obj, (bytes, bytearray)):
        if len(obj) > 400:
            return repr(obj[:400]) + "..."
        return repr(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _json_safe_eval_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe_eval_tree(x) for x in obj]
    if isinstance(obj, set):
        return sorted(str(x) for x in obj)
    return str(obj)


def _kernel_exec_result_to_json_dict(
    result: Any,
    *,
    log_prefix: str,
    tier: str,
) -> dict[str, Any]:
    """``KernelExecResult.model_dump(mode='json')``, with fallback if metadata holds
    non-JSON types (e.g. a live ``ImportError`` from extension load failures)."""
    from kernelbench.eval import KernelExecResult

    try:
        return result.model_dump(mode="json")
    except Exception as e:
        print(
            f"{log_prefix}   tier={tier} warning: model_dump(mode='json') failed "
            f"({type(e).__name__}: {e}); coercing non-JSON fields.",
            flush=True,
        )
        try:
            raw = result.model_dump(mode="python")
        except Exception as e2:
            print(
                f"{log_prefix}   tier={tier} ERROR: model_dump(mode='python') failed: {e2}",
                flush=True,
            )
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    "error": f"{type(e2).__name__}: {e2}",
                    "stress_tier": tier,
                    "serialization_fallback": True,
                },
            ).model_dump(mode="json")
        safe = _json_safe_eval_tree(raw)
        if not isinstance(safe, dict):
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    "error": "sanitized eval result was not a dict",
                    "stress_tier": tier,
                    "serialization_fallback": True,
                },
            ).model_dump(mode="json")
        try:
            return json.loads(json.dumps(safe, default=str))
        except (TypeError, ValueError) as e3:
            print(
                f"{log_prefix}   tier={tier} ERROR: json round-trip failed: {e3}",
                flush=True,
            )
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    "error": f"{type(e3).__name__}: {e3}",
                    "stress_tier": tier,
                    "serialization_fallback": True,
                },
            ).model_dump(mode="json")


def _recover_trajectory_after_stress_job_crash(
    job: dict[str, Any],
    *,
    new_run_name: str,
    script_tag: str,
    stress_refs_str: str,
    tiers: list[str],
    num_correct_trials: int,
    exc: BaseException,
    gpu_tag: str,
) -> None:
    """Avoid leaving ``in_progress`` if a job dies before the normal write."""
    from kernelbench.eval import KernelExecResult

    traj_path = Path(job["traj_path"])
    prefix = f"[stress{gpu_tag}]" if gpu_tag else "[stress]"
    try:
        with open(traj_path, encoding="utf-8") as f:
            traj = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"{prefix} could not read trajectory for crash recovery: {e}", flush=True)
        return

    err = f"{type(exc).__name__}: {exc}"
    merged = KernelExecResult(
        compiled=False,
        correctness=False,
        metadata={
            "error": err,
            "stress_job_crash": True,
        },
    ).model_dump(mode="json")
    merged["metadata"] = _stamp_metadata(
        dict(merged.get("metadata") or {}),
        script=script_tag,
        stress_refs=stress_refs_str,
    )
    traj["final_result"] = merged
    traj["outcome"] = _outcome_from_merged(merged)
    traj["run_name"] = new_run_name
    traj["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    traj["stress_eval"] = {
        "stress_refs": stress_refs_str,
        "tiers": list(tiers),
        "num_correct_trials_per_tier": int(num_correct_trials),
        "tier_errors": {"_job": err},
    }
    _write_trajectory_json(traj_path, traj)
    print(f"{prefix} wrote crash-recovery trajectory {traj_path.name}", flush=True)


def _init_dst_trajectories_after_copy(dst: Path, new_run_name: str) -> int:
    """Set ``run_name`` on all copied trajectories; clear ``correct`` outcomes and
    any prior ``stress_eval`` so static reports show pending until eval runs."""
    n_cleared = 0
    n_written = 0
    paths = sorted(dst.glob("**/*_trajectory.json"))
    for traj_path in paths:
        try:
            with open(traj_path, encoding="utf-8") as f:
                traj = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[stress] warning: could not read {traj_path}: {e}", flush=True)
            continue
        traj["run_name"] = new_run_name
        if traj.get("outcome") == "correct":
            traj["outcome"] = ""
            n_cleared += 1
        traj.pop("stress_eval", None)
        _write_trajectory_json(traj_path, traj)
        n_written += 1
    print(
        f"[stress] initialized {n_written} trajectory file(s) under {dst.name} "
        f"(run_name={new_run_name!r}; cleared outcome 'correct' on {n_cleared} "
        f"file(s) until stress re-eval completes).",
        flush=True,
    )
    return n_cleared


def _eval_with_retries(
    *,
    ref_arch_src: str,
    kernel_code: str,
    num_correct_trials: int,
    num_perf_trials: int,
    run_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    build_dir: str,
    device: Any,
    max_attempts: int = 5,
) -> Any:
    """Call ``eval_kernel_against_ref``; retry on None (JIT lock contention)."""
    import torch

    from kernelbench.eval import (
        KernelExecResult,
        eval_kernel_against_ref,
        get_torch_dtype_from_string,
    )

    backend = run_cfg["backend"]
    precision = _force_backend_precision(backend, run_cfg["precision"])
    torch_dtype = get_torch_dtype_from_string(precision)
    timing_method = str(run_cfg.get("timing_method", "cuda_event"))
    verbose = bool(agent_cfg.get("verbose", False))
    delay = 2.0
    last: KernelExecResult | None = None
    for attempt in range(max_attempts):
        last = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=kernel_code,
            seed_num=42,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            measure_performance=True,
            timing_method=timing_method,
            verbose=verbose,
            build_dir=build_dir,
            device=device,
            backend=backend,
            precision=torch_dtype,
            check_for_excessive_speedup=True,
        )
        if last is not None:
            return last
        print(
            f"[stress] eval returned None (lock?), retry {attempt + 1}/{max_attempts} "
            f"after {delay:.0f}s",
            flush=True,
        )
        time.sleep(delay)
        delay = min(delay * 1.5, 30.0)
    return last


def _patch_sweep_config_name(dst_run_dir: Path, new_name: str) -> None:
    cfg_path = dst_run_dir / "sweep_config.json"
    if not cfg_path.is_file():
        print(f"[stress] warning: no {cfg_path}; skip sweep_config patch", flush=True)
        return
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("run", {})["name"] = new_name
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def _stamp_metadata(meta: dict[str, Any], *, script: str, stress_refs: str) -> dict[str, Any]:
    out = dict(meta) if meta else {}
    out["stress_eval_script"] = script
    out["stress_eval_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out["stress_refs_root"] = stress_refs
    return out


def _cuda_fatal_in_tier_result(tier_result: dict[str, Any]) -> bool:
    """True if eval metadata indicates a CUDA context-fatal error."""
    meta = tier_result.get("metadata")
    if not isinstance(meta, dict):
        return False
    parts: list[str] = []
    for key in (
        "error",
        "compilation_error",
        "runtime_error",
        "compilation_error_name",
        "runtime_error_name",
    ):
        val = meta.get(key)
        if val is not None:
            parts.append(str(val))
    return _is_cuda_context_fatal(" ".join(parts))


def _skipped_tier_result(
    *,
    tier: str,
    reason: str,
    cuda_context_fatal: bool = False,
) -> dict[str, Any]:
    from kernelbench.eval import KernelExecResult

    meta: dict[str, Any] = {
        "error": reason,
        "stress_tier": tier,
        "stress_tier_skipped": True,
    }
    if cuda_context_fatal:
        meta["cuda_context_fatal"] = True
    return KernelExecResult(
        compiled=False,
        correctness=False,
        metadata=meta,
    ).model_dump(mode="json")


def _merge_tier_results(
    tier_results: dict[str, dict[str, Any]],
    *,
    num_correct_trials: int,
    tiers_attempted: list[str],
) -> dict[str, Any]:
    """Build one ``final_result``-shaped dict from per-tier ``KernelExecResult`` JSON."""
    from kernelbench.eval import KernelExecResult

    if not tier_results:
        return KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={"error": "stress_eval_no_tier_results"},
        ).model_dump(mode="json")

    compiled_all = all(tr.get("compiled") for tr in tier_results.values())
    correctness_all = all(tr.get("correctness") for tr in tier_results.values())

    # Prefer xl → awkward → large for top-level timing fields (compat with reports).
    runtime = -1.0
    ref_runtime = -1.0
    runtime_stats: dict[str, Any] = {}
    ref_runtime_stats: dict[str, Any] = {}
    for key in ("xl", "awkward", "large"):
        if key in tier_results and tier_results[key].get("compiled"):
            tr = tier_results[key]
            runtime = float(tr.get("runtime") or -1.0)
            ref_runtime = float(tr.get("ref_runtime") or -1.0)
            runtime_stats = dict(tr.get("runtime_stats") or {})
            ref_runtime_stats = dict(tr.get("ref_runtime_stats") or {})
            break

    meta: dict[str, Any] = {
        "stress_popcorn_eval": True,
        "stress_tiers": list(tier_results.keys()),
        "stress_tiers_order": tiers_attempted,
        "stress_num_correct_trials_per_tier": num_correct_trials,
        "stress_total_correctness_executions": num_correct_trials * len(tier_results),
        "stress_per_tier": tier_results,
    }

    last_tr = tier_results[tiers_attempted[-1]] if tiers_attempted else next(iter(tier_results.values()))
    out: dict[str, Any] = {
        "compiled": compiled_all,
        "correctness": correctness_all,
        "metadata": meta,
        "runtime": runtime,
        "runtime_stats": runtime_stats,
        "ref_runtime": ref_runtime,
        "ref_runtime_stats": ref_runtime_stats,
        "source_runtime": float(last_tr.get("source_runtime") or -1.0),
        "source_runtime_stats": dict(last_tr.get("source_runtime_stats") or {}),
        "source_backend": last_tr.get("source_backend"),
        "speedup_vs_source": float(last_tr.get("speedup_vs_source") or -1.0),
        "memory_stats": dict(last_tr.get("memory_stats") or {}),
        "numerical_precision": dict(last_tr.get("numerical_precision") or {}),
        "kernel_launch_stats": dict(last_tr.get("kernel_launch_stats") or {}),
        "sol_stats": dict(last_tr.get("sol_stats") or {}),
        "energy_stats": dict(last_tr.get("energy_stats") or {}),
        "roofline_stats": dict(last_tr.get("roofline_stats") or {}),
    }
    return out


def _run_one_kernel_job(
    *,
    job: dict[str, Any],
    device: Any,
    stress_root: Path,
    tiers: list[str],
    run_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    num_correct_trials: int,
    num_perf_trials: int,
    new_run_name: str,
    stress_refs_str: str,
    script_tag: str,
    gpu_tag: str = "",
) -> None:
    """Run all tiers for one kernel and write its trajectory JSON."""
    from kernelbench.agent.tools import _per_kernel_build_dir
    from kernelbench.eval import KernelExecResult

    kpath = Path(job["kpath"])
    traj_path = Path(job["traj_path"])
    lev = int(job["lev"])
    pid = int(job["pid"])
    problem_name = str(job["problem_name"])
    kernel_code = str(job["kernel_code"])
    variant = str(job["variant"])

    prefix = f"[stress{gpu_tag}]" if gpu_tag else "[stress]"

    print(
        f"{prefix} L{lev}/P{pid} name={problem_name} model_dir={kpath.parent.name}",
        flush=True,
    )
    try:
        with open(traj_path, encoding="utf-8") as f:
            traj_mark = json.load(f)
        traj_mark["outcome"] = "in_progress"
        traj_mark["run_name"] = new_run_name
        _write_trajectory_json(traj_path, traj_mark)
    except (OSError, json.JSONDecodeError) as e:
        print(f"{prefix} warning: could not mark in_progress: {e}", flush=True)

    t0 = time.perf_counter()
    tier_results: dict[str, dict[str, Any]] = {}
    tier_errors: dict[str, str] = {}
    cuda_context_aborted = False
    cuda_abort_after_tier: str | None = None

    model_dir = str(kpath.parent)
    cache_base = os.path.join(model_dir, f"level_{lev}_problem_{pid}_cache")

    for tier in tiers:
        if cuda_context_aborted:
            msg = (
                f"skipped: CUDA context fatal after tier {cuda_abort_after_tier!r}"
            )
            print(f"{prefix}   tier={tier} SKIP {msg}", flush=True)
            tier_errors[tier] = msg
            tier_results[tier] = _skipped_tier_result(
                tier=tier,
                reason=msg,
                cuda_context_fatal=True,
            )
            continue

        ref_path = stress_root / tier / f"level{lev}" / variant / problem_name
        if not ref_path.is_file():
            msg = f"missing ref file: {ref_path}"
            print(f"{prefix}   tier={tier} ERROR {msg}", flush=True)
            tier_errors[tier] = msg
            continue
        with open(ref_path, encoding="utf-8") as f:
            ref_src = f.read()

        tier_slug = f"{tier}_{hashlib.sha1(ref_src.encode()).hexdigest()[:8]}"
        build_dir = _per_kernel_build_dir(
            os.path.join(cache_base, "stress_popcorn", tier_slug),
            kernel_code,
        )
        os.makedirs(build_dir, exist_ok=True)

        try:
            result = _eval_with_retries(
                ref_arch_src=ref_src,
                kernel_code=kernel_code,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials,
                run_cfg=run_cfg,
                agent_cfg=agent_cfg,
                build_dir=build_dir,
                device=device,
            )
        except Exception as e:
            print(f"{prefix}   tier={tier} EXCEPTION {e}", flush=True)
            traceback.print_exc()
            tier_errors[tier] = f"{type(e).__name__}: {e}"
            result = None
            if _is_cuda_context_fatal(str(e)) or _is_cuda_context_fatal(
                traceback.format_exc()
            ):
                cuda_context_aborted = True
                cuda_abort_after_tier = tier
                print(
                    f"{prefix}   CUDA context fatal on tier={tier}; "
                    f"skipping remaining tiers in this job.",
                    flush=True,
                )

        if result is None:
            tier_errors.setdefault(tier, "eval returned None after retries")
            tier_results[tier] = KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    "error": tier_errors.get(tier, "eval_failed"),
                    "stress_tier": tier,
                },
            ).model_dump(mode="json")
        else:
            tier_results[tier] = _kernel_exec_result_to_json_dict(
                result, log_prefix=prefix, tier=tier
            )

        tr = tier_results[tier]
        if not cuda_context_aborted and _cuda_fatal_in_tier_result(tr):
            cuda_context_aborted = True
            cuda_abort_after_tier = tier
            print(
                f"{prefix}   CUDA context fatal detected in tier={tier} result; "
                f"skipping remaining tiers in this job.",
                flush=True,
            )

        print(
            f"{prefix}   tier={tier} compiled={tr.get('compiled')} "
            f"correctness={tr.get('correctness')}",
            flush=True,
        )

    merged = _merge_tier_results(
        tier_results,
        num_correct_trials=num_correct_trials,
        tiers_attempted=list(tiers),
    )
    if isinstance(merged.get("metadata"), dict):
        merged["metadata"] = _stamp_metadata(
            merged["metadata"],
            script=script_tag,
            stress_refs=stress_refs_str,
        )
        if tier_errors:
            merged["metadata"]["stress_tier_errors"] = tier_errors
        if cuda_context_aborted and cuda_abort_after_tier:
            merged["metadata"]["stress_cuda_context_aborted_after"] = (
                cuda_abort_after_tier
            )
        missing = [t for t in tiers if t not in tier_results]
        if missing:
            merged["compiled"] = False
            merged["correctness"] = False
            merged["metadata"]["stress_missing_tiers"] = missing

    with open(traj_path, encoding="utf-8") as f:
        traj = json.load(f)
    traj["final_result"] = merged
    traj["outcome"] = _outcome_from_merged(merged)
    traj["run_name"] = new_run_name
    traj["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stress_eval: dict[str, Any] = {
        "stress_refs": stress_refs_str,
        "tiers": tiers,
        "num_correct_trials_per_tier": int(num_correct_trials),
        "tier_errors": tier_errors,
    }
    if cuda_context_aborted and cuda_abort_after_tier:
        stress_eval["cuda_context_aborted_after"] = cuda_abort_after_tier
    traj["stress_eval"] = stress_eval
    _write_trajectory_json(traj_path, traj)

    dt = time.perf_counter() - t0
    print(
        f"{prefix}   -> outcome={traj['outcome']} compiled={merged.get('compiled')} "
        f"correctness={merged.get('correctness')} ({dt:.1f}s wall)",
        flush=True,
    )


def _run_one_kernel_job_isolated_entry(job_bundle: dict[str, Any]) -> None:
    """Spawn target: one kernel job in a fresh process (clean CUDA context)."""
    gpu_id = int(job_bundle["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch as _torch

    from kernelbench.utils import set_gpu_arch

    if not _torch.cuda.is_available():
        print(
            f"[stress gpu{gpu_id}] CUDA not available; cannot run "
            f"L{job_bundle['job']['lev']}/P{job_bundle['job']['pid']}.",
            flush=True,
        )
        raise SystemExit(1)

    device = _torch.device("cuda:0")
    _torch.cuda.set_device(device)

    common = job_bundle["common"]
    set_gpu_arch(common["gpu_arch"])
    job = job_bundle["job"]
    gpu_tag = f" gpu{gpu_id}"

    try:
        _run_one_kernel_job(
            job=job,
            device=device,
            stress_root=Path(common["stress_root"]),
            tiers=list(common["tiers"]),
            run_cfg=common["run_cfg"],
            agent_cfg=common["agent_cfg"],
            num_correct_trials=int(common["num_correct_trials"]),
            num_perf_trials=int(common["num_perf_trials"]),
            new_run_name=str(common["new_run_name"]),
            stress_refs_str=str(common["stress_refs_str"]),
            script_tag=str(common["script_tag"]),
            gpu_tag=gpu_tag,
        )
    except Exception as e:
        print(
            f"[stress{gpu_tag}] job crash L{job['lev']}/P{job['pid']}: {e}",
            flush=True,
        )
        traceback.print_exc()
        _recover_trajectory_after_stress_job_crash(
            job,
            new_run_name=str(common["new_run_name"]),
            script_tag=str(common["script_tag"]),
            stress_refs_str=str(common["stress_refs_str"]),
            tiers=list(common["tiers"]),
            num_correct_trials=int(common["num_correct_trials"]),
            exc=e,
            gpu_tag=gpu_tag,
        )
        raise SystemExit(1) from e


def _dispatch_kernel_job_isolated(
    *,
    job: dict[str, Any],
    gpu_id: int,
    common: dict[str, Any],
    mp_ctx: mp.context.BaseContext,
) -> bool:
    """Run one kernel job in a child process. Returns True if the job failed."""
    job_bundle = {"gpu_id": gpu_id, "job": job, "common": common}
    proc = mp_ctx.Process(
        target=_run_one_kernel_job_isolated_entry,
        args=(job_bundle,),
    )
    proc.start()
    proc.join()
    if proc.exitcode == 0:
        return False
    print(
        f"[stress gpu{gpu_id}] isolated job L{job['lev']}/P{job['pid']} "
        f"exitcode={proc.exitcode}",
        flush=True,
    )
    return True


def _gpu_worker_process(bundle: dict[str, Any]) -> None:
    """Child process: pin one physical GPU, drain jobs (each in its own subprocess)."""
    gpu_id = int(bundle["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch as _torch

    if not _torch.cuda.is_available():
        print(
            f"[stress gpu{gpu_id}] CUDA not available; skipping "
            f"{len(bundle['jobs'])} jobs.",
            flush=True,
        )
        raise SystemExit(1)

    common = bundle["common"]
    mp_ctx = mp.get_context("spawn")
    job_failures = common.get("job_failures")

    for job in bundle["jobs"]:
        failed = _dispatch_kernel_job_isolated(
            job=job,
            gpu_id=gpu_id,
            common=common,
            mp_ctx=mp_ctx,
        )
        if failed and job_failures is not None:
            with job_failures.get_lock():
                job_failures.value += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src-run",
        required=True,
        help="Source run directory relative to repo root (or absolute), "
        "must contain sweep_config.json.",
    )
    ap.add_argument(
        "--dst-run-name",
        required=True,
        help="Base directory name under runs/. A timestamp suffix is appended by "
        "default so retries do not collide with a partial/broken destination; "
        "see --dst-exact-name.",
    )
    ap.add_argument(
        "--dst-exact-name",
        action="store_true",
        help="Use --dst-run-name verbatim as runs/<name> (must not exist). "
        "Default is to append _YYYYMMDD_HHMMSS_microseconds.",
    )
    ap.add_argument(
        "--stress-refs",
        required=True,
        help="Root with subdirs large/, awkward/, xl/ mirroring levelN/<variant>/*.py",
    )
    ap.add_argument(
        "--tiers",
        default="large,awkward,xl",
        help="Comma-separated tier directory names under --stress-refs (default: large,awkward,xl).",
    )
    ap.add_argument(
        "--num-correct-trials",
        type=int,
        default=5,
        help="Per-tier correctness trials (default 5 → 15 total for 3 tiers).",
    )
    ap.add_argument(
        "--num-perf-trials",
        type=int,
        default=5,
        help="Perf trials per tier eval (default 5; keep moderate for large tensors).",
    )
    ap.add_argument(
        "--include-l2-comm",
        action="store_true",
        help="Also stress-eval L2 popcorn multi-GPU problem ids (2,11,18,27,34,38); "
        "default is to skip them on single-GPU nodes.",
    )
    ap.add_argument(
        "--build-report",
        action="store_true",
        help="Run scripts/build_report.py on the destination run after stress eval.",
    )
    ap.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of parallel worker processes (default 8). Capped by visible GPUs.",
    )
    ap.add_argument(
        "--serial",
        action="store_true",
        help="Single-process eval on cuda:0 only (no multiprocessing).",
    )
    args = ap.parse_args()

    import torch

    from kernelbench.utils import set_gpu_arch

    if not torch.cuda.is_available():
        print("[stress] ERROR: CUDA not available.", flush=True)
        return 1

    src = Path(_resolve_under_repo(args.src_run))
    if not src.is_dir():
        print(f"[stress] ERROR: source run directory does not exist: {src}", flush=True)
        return 1
    cfg_src = src / "sweep_config.json"
    if not cfg_src.is_file():
        print(f"[stress] ERROR: missing {cfg_src}", flush=True)
        return 1

    if args.dst_exact_name:
        dst_run_name = args.dst_run_name
    else:
        dst_run_name = (
            f"{args.dst_run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        )
    dst = Path(REPO_TOP) / "runs" / dst_run_name
    if dst.exists():
        print(f"[stress] ERROR: destination already exists: {dst}", flush=True)
        return 1
    if not args.dst_exact_name:
        print(
            f"[stress] destination run dir (timestamped): {dst_run_name!r} "
            f"(base {args.dst_run_name!r})",
            flush=True,
        )

    stress_root = Path(_resolve_under_repo(args.stress_refs))
    if not stress_root.is_dir():
        print(f"[stress] ERROR: --stress-refs is not a directory: {stress_root}", flush=True)
        return 1

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    if not tiers:
        print("[stress] ERROR: empty --tiers", flush=True)
        return 1
    for tier in tiers:
        p = stress_root / tier
        if not p.is_dir():
            print(f"[stress] ERROR: missing tier directory: {p}", flush=True)
            return 1

    print(f"[stress] copytree\n  {src}\n  -> {dst}", flush=True)
    shutil.copytree(src, dst)

    new_run_name = dst_run_name
    _patch_sweep_config_name(dst, new_run_name)
    _init_dst_trajectories_after_copy(dst, new_run_name)

    with open(dst / "sweep_config.json", encoding="utf-8") as f:
        sweep_cfg = json.load(f)
    run_cfg = sweep_cfg["run"]
    agent_cfg = sweep_cfg.get("agent", {})

    if run_cfg.get("prompt_option") == "hardware_translation":
        print(
            "[stress] ERROR: this script is for popcorn-style sweeps "
            "(problem.code / local variant py files). "
            "Use reval_saved_torchrun_sweep.py for hardware_translation.",
            flush=True,
        )
        shutil.rmtree(dst, ignore_errors=True)
        return 1

    variants = run_cfg.get("variants") or ["popcorn"]
    if len(variants) != 1:
        print(
            f"[stress] WARNING: expected a single variant in sweep_config; got {variants}. "
            f"Using variants[0]={variants[0]!r}.",
            flush=True,
        )
    variant = str(variants[0])

    gpu_arch = run_cfg.get("gpu_arch", ["Hopper"])
    ga = gpu_arch if isinstance(gpu_arch, list) else [gpu_arch]

    levels = [int(x) for x in run_cfg.get("levels", [1, 2, 3])]
    datasets: dict[int, Any] = {}
    for lev in levels:
        datasets[lev] = construct_kernelbench_dataset(
            level=lev,
            source=run_cfg.get("dataset_src", "local"),
            dataset_name=run_cfg.get("dataset_name", "ScalingIntelligence/KernelBench"),
            variant=variant,
        )

    kernel_files = sorted(dst.glob("**/level_*_problem_*_kernel.py"))
    script_tag = os.path.basename(sys.argv[0])
    stress_refs_str = str(stress_root.resolve())

    jobs: list[dict[str, Any]] = []
    for kpath in kernel_files:
        m = _KERNEL_PATH_RE.search(kpath.name)
        if not m:
            print(f"[stress] skip (unexpected name): {kpath}", flush=True)
            continue
        lev, pid = int(m.group(1)), int(m.group(2))
        traj_path = kpath.with_name(kpath.name.replace("_kernel.py", "_trajectory.json"))
        if not traj_path.is_file():
            print(f"[stress] skip (no trajectory): {kpath}", flush=True)
            continue

        if (
            not args.include_l2_comm
            and lev == 2
            and variant == "popcorn"
            and pid in L2_POPCORN_MULTI_GPU_IDS
        ):
            print(
                f"[stress] skip L2/popcorn multi-GPU problem_id={pid}: {kpath.name}",
                flush=True,
            )
            continue

        with open(kpath, encoding="utf-8") as f:
            kernel_code = f.read()
        if not kernel_code.strip():
            print(f"[stress] skip (empty kernel): {kpath}", flush=True)
            continue

        if lev not in datasets:
            print(f"[stress] skip (level {lev} not in sweep levels): {kpath}", flush=True)
            continue

        try:
            problem = datasets[lev].get_problem_by_id(pid)
        except ValueError as e:
            print(f"[stress] skip (dataset): {e}", flush=True)
            continue

        jobs.append(
            {
                "kpath": str(kpath.resolve()),
                "traj_path": str(traj_path.resolve()),
                "lev": lev,
                "pid": pid,
                "problem_name": problem.name,
                "kernel_code": kernel_code,
                "variant": variant,
            }
        )

    visible = int(torch.cuda.device_count())
    if args.serial:
        num_workers = 1
    else:
        num_workers = min(max(1, int(args.num_gpus)), visible, max(1, len(jobs)))

    mode = "serial" if num_workers == 1 else f"parallel num_gpus={num_workers} (visible={visible})"
    print(
        f"[stress] tiers={tiers} num_correct_trials={args.num_correct_trials} "
        f"num_perf_trials={args.num_perf_trials} stress_refs={stress_refs_str}\n"
        f"[stress] {mode}: {len(jobs)} kernel jobs (from {len(kernel_files)} kernel files)",
        flush=True,
    )

    if not jobs:
        print("[stress] WARNING: no kernel jobs to evaluate.", flush=True)
        have_failures = False
    elif num_workers == 1:
        have_failures = False
        common = {
            "stress_root": str(stress_root.resolve()),
            "tiers": tiers,
            "run_cfg": run_cfg,
            "agent_cfg": agent_cfg,
            "num_correct_trials": int(args.num_correct_trials),
            "num_perf_trials": int(args.num_perf_trials),
            "new_run_name": new_run_name,
            "stress_refs_str": stress_refs_str,
            "script_tag": script_tag,
            "gpu_arch": ga,
        }
        mp_ctx = mp.get_context("spawn")
        for job in jobs:
            failed = _dispatch_kernel_job_isolated(
                job=job,
                gpu_id=0,
                common=common,
                mp_ctx=mp_ctx,
            )
            if failed:
                have_failures = True
    else:
        job_failures = mp.Value("i", 0)
        worker_failed = False
        common = {
            "stress_root": str(stress_root.resolve()),
            "tiers": tiers,
            "run_cfg": run_cfg,
            "agent_cfg": agent_cfg,
            "num_correct_trials": int(args.num_correct_trials),
            "num_perf_trials": int(args.num_perf_trials),
            "new_run_name": new_run_name,
            "stress_refs_str": stress_refs_str,
            "script_tag": script_tag,
            "gpu_arch": ga,
            "job_failures": job_failures,
        }
        chunks: list[list[dict[str, Any]]] = [[] for _ in range(num_workers)]
        for i, job in enumerate(jobs):
            chunks[i % num_workers].append(job)

        ctx = mp.get_context("spawn")
        procs: list[mp.Process] = []
        for gid in range(num_workers):
            if not chunks[gid]:
                continue
            bundle = {"gpu_id": gid, "jobs": chunks[gid], "common": common}
            p = ctx.Process(target=_gpu_worker_process, args=(bundle,))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                worker_failed = True
                print(
                    f"[stress] WARNING: worker pid={p.pid} exitcode={p.exitcode}",
                    flush=True,
                )
        have_failures = worker_failed or (job_failures.value > 0)
        if job_failures.value > 0:
            print(
                f"[stress] ERROR: {job_failures.value} kernel job(s) raised before completion.",
                flush=True,
            )

    if have_failures:
        print(
            "[stress] ERROR: stress re-eval finished with worker and/or job failures "
            "(see logs above).",
            flush=True,
        )

    if args.build_report:
        br = os.path.join(REPO_TOP, "scripts", "build_report.py")
        cmd = [sys.executable, br, str(dst)]
        print(f"[stress] running: {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd=REPO_TOP)
        if r.returncode != 0:
            print(f"[stress] build_report exited {r.returncode}", flush=True)
            return r.returncode

    print(f"[stress] done. Destination: {dst}", flush=True)
    return 1 if have_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
