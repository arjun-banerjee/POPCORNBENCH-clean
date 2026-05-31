"""Multi-tier popcorn stress evaluation (large / awkward / xl) for sweep eval."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Sequence

from kernelbench.agent.eval_server import _is_cuda_context_fatal

DEFAULT_STRESS_TIERS: tuple[str, ...] = ("large", "awkward", "xl")
DEFAULT_STRESS_REFS_ROOT = "KernelBench/stress_refs2"


def resolve_stress_ref_path(
    stress_root: str | Path,
    tier: str,
    level: int,
    variant: str,
    problem_name: str,
) -> Path:
    return Path(stress_root) / tier / f"level{level}" / variant / problem_name


def merge_tier_kernel_exec_results(
    tier_results: dict[str, Any],
    *,
    num_correct_trials: int,
    tiers_attempted: Sequence[str],
) -> dict[str, Any]:
    """Build one ``KernelExecResult``-shaped dict from per-tier results."""
    from kernelbench.eval import KernelExecResult

    if not tier_results:
        return KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={"error": "stress_eval_no_tier_results"},
        ).model_dump(mode="json")

    compiled_all = all(tr.get("compiled") for tr in tier_results.values())
    correctness_all = all(tr.get("correctness") for tr in tier_results.values())

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

    n_tiers = len(tier_results)
    total_exec = int(num_correct_trials) * n_tiers
    meta: dict[str, Any] = {
        "stress_popcorn_eval": True,
        "stress_tiers": list(tier_results.keys()),
        "stress_tiers_order": list(tiers_attempted),
        "stress_num_correct_trials_per_tier": int(num_correct_trials),
        "stress_total_correctness_executions": total_exec,
        "stress_per_tier": tier_results,
        "correctness_trials": (
            f"stress {num_correct_trials}/tier × {n_tiers} tiers ({total_exec} execs)"
        ),
    }

    last_tr = (
        tier_results[tiers_attempted[-1]]
        if tiers_attempted
        else next(iter(tier_results.values()))
    )
    return {
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


def _kernel_result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return dict(result)


def eval_kernel_stress_tiers(
    *,
    kernel_code: str,
    level: int,
    variant: str,
    problem_name: str,
    stress_refs_root: str | Path,
    tiers: Sequence[str] = DEFAULT_STRESS_TIERS,
    num_correct_trials: int,
    num_perf_trials: int = 0,
    measure_performance: bool = True,
    build_dir: str | None,
    device: Any,
    backend: str,
    precision: Any,
    timing_method: str = "cuda_event",
    verbose: bool = False,
    check_for_excessive_speedup: bool = True,
    seed_num: int = 42,
) -> Any | None:
    """
    Run ``eval_kernel_against_ref`` once per stress tier (separate reference trees).

    Returns merged ``KernelExecResult`` or None on persistent lock contention.
    """
    from kernelbench.agent.tools import _per_kernel_build_dir
    from kernelbench.eval import KernelExecResult, eval_kernel_against_ref

    stress_root = Path(stress_refs_root)
    tier_results: dict[str, dict[str, Any]] = {}
    cuda_context_aborted = False
    cuda_abort_after: str | None = None

    for tier in tiers:
        if cuda_context_aborted:
            tier_results[tier] = KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    "error": (
                        f"skipped: CUDA context fatal after tier {cuda_abort_after!r}"
                    ),
                    "stress_tier": tier,
                    "stress_tier_skipped": True,
                    "cuda_context_fatal": True,
                },
            ).model_dump(mode="json")
            continue

        ref_path = resolve_stress_ref_path(
            stress_root, tier, level, variant, problem_name
        )
        if not ref_path.is_file():
            tier_results[tier] = KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    "error": f"missing stress ref: {ref_path}",
                    "stress_tier": tier,
                },
            ).model_dump(mode="json")
            continue

        ref_src = ref_path.read_text(encoding="utf-8")
        tier_slug = f"{tier}_{hashlib.sha1(ref_src.encode()).hexdigest()[:8]}"
        tier_build = _per_kernel_build_dir(
            os.path.join(build_dir or ".", "stress_popcorn", tier_slug),
            kernel_code,
        )
        os.makedirs(tier_build, exist_ok=True)

        result = eval_kernel_against_ref(
            original_model_src=ref_src,
            custom_model_src=kernel_code,
            seed_num=seed_num,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            measure_performance=measure_performance,
            timing_method=timing_method,
            verbose=verbose,
            build_dir=tier_build,
            device=device,
            backend=backend,
            precision=precision,
            check_for_excessive_speedup=check_for_excessive_speedup,
        )
        if result is None:
            return None

        tr = _kernel_result_to_dict(result)
        tier_results[tier] = tr
        meta = tr.get("metadata") or {}
        fatal_parts = [
            str(meta.get(k, ""))
            for k in (
                "error",
                "compilation_error",
                "runtime_error",
                "compilation_error_name",
                "runtime_error_name",
            )
        ]
        if _is_cuda_context_fatal(" ".join(fatal_parts)):
            cuda_context_aborted = True
            cuda_abort_after = tier

    merged = merge_tier_kernel_exec_results(
        tier_results,
        num_correct_trials=num_correct_trials,
        tiers_attempted=list(tiers),
    )
    missing = [t for t in tiers if t not in tier_results]
    if missing:
        merged["compiled"] = False
        merged["correctness"] = False
        meta = dict(merged.get("metadata") or {})
        meta["stress_missing_tiers"] = missing
        merged["metadata"] = meta

    from kernelbench.eval import KernelExecResult

    return KernelExecResult.model_validate(merged)
