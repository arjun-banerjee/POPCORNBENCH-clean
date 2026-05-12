"""
Tool definitions and executors for the KernelBench agent.

Design principles
-----------------
- Each tool is a self-contained class with a clear input/output contract.
- Tools are synchronous. The interface is async-ready — wrap execute() calls
  in asyncio.to_thread if needed.
- input_schema follows JSON Schema (same format as MCP / OpenAI function
  calling), so wrapping any tool in an MCP server later is a thin transport
  layer only.
- ToolContext carries shared per-run state (problem source, device, backend,
  etc.) so tools don't need globals.

Tool catalogue
--------------
    compile_kernel    — try to compile; return compiler output
    run_correctness   — correctness trials only (no timing)
    profile_kernel    — nsight roofline profiling (opt-in, requires ncu)
    get_gpu_specs     — hardware specs for the current device
    static_check      — static reward-hack pattern detector
    submit_kernel     — full eval: correctness + timing (speedup not revealed)

Output format
-------------
Every tool's `output` string follows the rule:

    {ToolName} {PASSED|FAILED}: {one-line summary}
    {optional detail block}

Two tools are exceptions to the PASS/FAIL rule — they have no natural
success/failure framing:

    - get_gpu_specs : reference data only; starts with "GPU specs for {name}:"
    - static_check  : uses PASSED / PASSED (with warnings) / FAILED to
                      distinguish clean / warnings-only / strict-violation runs.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Callable

import torch

from kernelbench.distributed_torchrun_eval import (
    eval_kernel_via_torchrun,
    reference_uses_torchrun_collectives,
)
from kernelbench.eval import (
    KernelExecResult,
    eval_kernel_against_ref,
    load_custom_model,
    load_custom_model_with_tempfile,
    graceful_eval_cleanup,
    get_torch_dtype_from_string,
)
from kernelbench.kernel_static_checker import validate_kernel_static
from kernelbench.reference_timing import (
    compute_dynamic_timeout,
    probe_reference_runtime,
)


# When eval_kernel_against_ref returns None (e.g. torch JIT extension lock-file
# contention from oversubscribed agents on the same GPU), retry transparently
# instead of reporting failure to the model — it's our infra, not a bad kernel.
_LOCK_RETRY_ATTEMPTS = 8
_LOCK_RETRY_BASE_SLEEP_S = 0.5  # exponential backoff with jitter


def _per_kernel_build_dir(base_build_dir: str | None, kernel_code: str) -> str | None:
    """Return a content-keyed subdirectory of base_build_dir.

    Same kernel → same subdir → torch JIT cache hit (fast).
    Different kernel → different subdir → no shared lock files, no contention.

    Returns None if base_build_dir is None (caller falls back to default cache).
    """
    if not base_build_dir:
        return None
    digest = hashlib.sha1(kernel_code.encode("utf-8", errors="replace")).hexdigest()[:12]
    sub = os.path.join(base_build_dir, f"k_{digest}")
    os.makedirs(sub, exist_ok=True)
    return sub


def _retry_eval_on_lock(
    eval_fn: Callable[[], Any],
    *,
    build_dir: str | None = None,
) -> Any:
    """Call eval_fn(); if it returns None, retry with exponential backoff.

    `eval_kernel_against_ref` returns None when it hits a lock-file or
    "No such file or directory" error during cpp_extension build. These
    can be transient (truly concurrent builds) or persistent (stale state
    left by a crashed earlier process — most common after we respawn an
    eval server on a CUDA-fatal kernel and the build dir is half-written).

    To recover from the persistent case, we wipe the build dir between
    retries so the next attempt starts from a clean slate. The dir is
    re-created lazily by torch.utils.cpp_extension on the next build.
    """
    import shutil
    for attempt in range(_LOCK_RETRY_ATTEMPTS):
        result = eval_fn()
        if result is not None:
            return result
        if attempt == _LOCK_RETRY_ATTEMPTS - 1:
            return None
        # Best-effort wipe of any stale build state. Safe to fail (e.g. if
        # a sibling process is currently writing to the dir).
        if build_dir and os.path.isdir(build_dir):
            try:
                shutil.rmtree(build_dir)
                os.makedirs(build_dir, exist_ok=True)
            except OSError:
                pass
        sleep_s = _LOCK_RETRY_BASE_SLEEP_S * (2 ** attempt) + random.uniform(0, 0.25)
        time.sleep(sleep_s)
    return None


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg or "cudaerrormemoryallocation" in msg


def _run_with_oom_retry(fn: Callable[[], Any], *, max_retries: int = 2) -> Any:
    """Run fn, catching CUDA OOM and retrying after empty_cache + gc.

    Concurrent agents on the same GPU can transiently push memory pressure
    over the line. Most OOMs clear if we drop cached blocks and yield for
    a moment. After max_retries the underlying exception is re-raised so
    the caller can record a real failure.
    """
    import gc
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001
            if not _is_cuda_oom(exc):
                raise
            last_exc = exc
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            sleep_s = 0.5 * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Dynamic eval-timeout resolution
# ---------------------------------------------------------------------------


def _resolve_dynamic_eval_timeouts(ctx) -> tuple[int, int]:
    """Return ``(dyn_correctness_s, dyn_submit_s)`` for the current context.

    - When ``ctx.dynamic_eval_timeout`` is False, both equal
      ``ctx.eval_torchrun_timeout_s`` (the configured static ceiling).
    - Otherwise, probe the reference runtime once per
      (level, problem_id, world_size) and apply the documented formulas.
      Probe failure falls back to ``ctx.eval_torchrun_timeout_s``.

    Side effects:
    - Caches ``t_ref`` on ``ctx._ref_runtime_cache`` keyed by
      (level, problem_id, world_size).
    - Logs the probe result and resolved timeouts the first time we resolve
      a given (level, problem_id, world_size).
    """
    ceiling = int(ctx.eval_torchrun_timeout_s or 3600)
    if not getattr(ctx, "dynamic_eval_timeout", False):
        return ceiling, ceiling

    ws = max(1, int(getattr(ctx, "distributed_torchrun_world_size", 1) or 1))
    level = int(getattr(ctx, "level", 0))
    pid = int(getattr(ctx, "problem_id", 0))
    cache = ctx._ref_runtime_cache if ctx._ref_runtime_cache is not None else {}
    logged = ctx._probe_logged if ctx._probe_logged is not None else {}
    key = (level, pid, ws)

    if key in cache:
        t_ref = cache[key]
    else:
        # Probe with a small trial count — we only need an order-of-magnitude
        # estimate. Reuses correctness's trial count cap so the probe wall
        # time is comparable to a single correctness pass.
        probe_trials = max(1, min(int(ctx.num_correct_trials or 5), 5))
        t_ref = probe_reference_runtime(
            ref_arch_src=ctx.ref_arch_src,
            world_size=ws,
            num_trials=probe_trials,
            backend=ctx.backend,
            precision_str=ctx.precision,
            device=getattr(ctx, "device", None),
            probe_timeout_s=int(getattr(ctx, "reference_probe_timeout_s", 300) or 300),
            build_dir=getattr(ctx, "build_dir", None),
            verbose=bool(getattr(ctx, "verbose", False)),
        )
        cache[key] = t_ref  # may be None on failure; cached so we don't retry

    dyn_correctness = compute_dynamic_timeout(
        t_ref_s=t_ref,
        overhead_s=float(getattr(ctx, "correctness_overhead_s", 120) or 0),
        k=float(getattr(ctx, "correctness_timeout_k", 10.0) or 0),
        n_trials=int(ctx.num_correct_trials or 0),
        floor_s=float(getattr(ctx, "correctness_floor_s", 120) or 0),
        ceiling_s=ceiling,
    )
    dyn_submit = compute_dynamic_timeout(
        t_ref_s=t_ref,
        overhead_s=float(getattr(ctx, "submit_overhead_s", 180) or 0),
        k=float(getattr(ctx, "submit_timeout_k", 10.0) or 0),
        n_trials=int((ctx.num_correct_trials or 0) + (ctx.num_perf_trials or 0)),
        floor_s=float(getattr(ctx, "submit_floor_s", 300) or 0),
        ceiling_s=ceiling,
    )

    if not logged.get(key):
        logged[key] = True
        if t_ref is None:
            print(
                f"[L{level}/P{pid}] reference probe FAILED (world={ws}); "
                f"using ceiling {ceiling}s for run_correctness + submit_kernel",
                flush=True,
            )
        else:
            print(
                f"[L{level}/P{pid}] reference probe: t_ref={t_ref:.3f}s "
                f"(world={ws}) -> dyn_correctness={dyn_correctness}s, "
                f"dyn_submit={dyn_submit}s (cap {ceiling}s)",
                flush=True,
            )

    return dyn_correctness, dyn_submit


# ---------------------------------------------------------------------------
# ToolContext — shared per-problem state passed into every tool
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Shared per-run state passed into every tool call."""

    ref_arch_src: str  # reference PyTorch model source
    backend: str  # "cuda" | "triton" | "tilelang" | "cute" | "hip"
    precision: str  # "fp32" | "fp16" | "bf16"
    device: torch.device  # GPU device to run on
    build_dir: str | None = None  # CUDA compile cache directory
    num_correct_trials: int = 5  # correctness trials in submit_kernel
    num_perf_trials: int = 100  # timing trials in submit_kernel
    timing_method: str = "cuda_event"
    verbose: bool = False
    # Optional: when set, submit_kernel (and optionally profile_kernel) RPC
    # to a per-GPU eval server instead of running locally. Decouples eval
    # serialization from agent worker oversubscription. See eval_client.py.
    eval_client: Any = None

    # When >1 and ``ref_arch_src`` imports ``kernelbench.distributed_collectives``,
    # submit_kernel runs eval via ``torchrun`` with all GPUs (CUDA_VISIBLE_DEVICES
    # cleared for the subprocess). Default 8 for L5 multi-GPU comm benchmarks.
    distributed_torchrun_world_size: int = 1
    eval_torchrun_timeout_s: int = 3600

    # --- Dynamic eval-timeout knobs (see kernelbench.reference_timing). -----
    # When ``dynamic_eval_timeout`` is True, RunCorrectnessTool / SubmitKernelTool
    # probe the reference forward time once per (level, problem_id, world_size)
    # via ``probe_reference_runtime`` and derive per-tool caps from the
    # formulas below; ``eval_torchrun_timeout_s`` stays the hard ceiling.
    dynamic_eval_timeout: bool = False
    correctness_overhead_s: int = 120
    correctness_timeout_k: float = 10.0
    correctness_floor_s: int = 120
    submit_overhead_s: int = 180
    submit_timeout_k: float = 10.0
    submit_floor_s: int = 300
    reference_probe_timeout_s: int = 300
    # Identity used for the probe cache key (set by run_sweep.py / KernelAgent).
    level: int = 0
    problem_id: int = 0
    # Shared mutable handle for the per-worker probe cache. Initialized to
    # an empty dict in __post_init__ unless caller injects their own; tools
    # populate it lazily on first call. Keyed by (level, problem_id, world_size).
    _ref_runtime_cache: dict | None = None
    # Per-(level, problem_id, world_size) "we already logged the probe" flag.
    _probe_logged: dict | None = None

    # Mutable: last profile result for delta comparison across iterations
    _last_profile_summary: Any = field(default=None, init=False, repr=False)
    # Multi-rank torchrun profile state: rank_id -> ProfileSummary. Updated
    # only by ProfileKernelTool when ctx.distributed_torchrun_world_size > 1.
    _last_per_rank_profile_summary: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Ensure shared mutable holders exist so callers can rely on them.
        if self._ref_runtime_cache is None:
            self._ref_runtime_cache = {}
        if self._probe_logged is None:
            self._probe_logged = {}

    @property
    def torch_precision(self) -> torch.dtype:
        return get_torch_dtype_from_string(self.precision)


# ---------------------------------------------------------------------------
# ToolResult — output of one tool execution
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str  # LLM-readable text
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool ABC
# ---------------------------------------------------------------------------


class Tool(ABC):
    """Abstract base class for all agent tools."""

    name: str
    description: str
    # JSON Schema for the tool's input parameters.
    # Tools that take no arguments (get_gpu_specs) use an empty-properties schema.
    input_schema: dict[str, Any]

    @abstractmethod
    def execute(self, ctx: ToolContext, **kwargs) -> ToolResult: ...

    def to_responses_schema(self) -> dict[str, Any]:
        """
        OpenAI Responses-API tool schema (flat shape):

            {"type": "function", "name": ..., "description": ..., "parameters": ...}

        This is the shape consumed by `client.responses.create(tools=[...])`.
        It differs from the Chat Completions schema, which nests fields under
        a "function" key.
        """
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }

    def to_mcp_schema(self) -> dict[str, Any]:
        """MCP tool schema (different envelope, same schema content)."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


# ---------------------------------------------------------------------------
# Shared input-schema fragment for kernel_code
# ---------------------------------------------------------------------------

_KERNEL_CODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kernel_code": {
            "type": "string",
            "description": (
                "Full Python source of the ModelNew kernel file. Must be a "
                "complete, valid Python module — not raw CUDA C/C++."
            ),
        }
    },
    "required": ["kernel_code"],
}


# ---------------------------------------------------------------------------
# 1. CompileKernelTool
# ---------------------------------------------------------------------------


class CompileKernelTool(Tool):
    name = "compile_kernel"
    description = (
        "Compile the kernel without running it. Use this first after writing "
        "or editing a kernel to catch syntax, linker, and CUDA-compilation "
        "errors cheaply before spending GPU time on correctness. Returns "
        "compiler output on failure, or a success confirmation."
    )
    input_schema = _KERNEL_CODE_SCHEMA

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        # When the agent has an eval RPC client, route compile to the
        # per-GPU server so the agent process never initializes CUDA.
        if ctx.eval_client is not None:
            return ctx.eval_client.compile_kernel(ctx, kernel_code)

        stdout_buf = StringIO()
        context: dict = {}
        build_dir = _per_kernel_build_dir(ctx.build_dir, kernel_code)

        try:
            os.environ["TORCH_USE_CUDA_DSA"] = "1"
            torch.cuda.set_device(ctx.device)

            with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
                backend_lower = ctx.backend.lower()
                if backend_lower in ("triton", "tilelang", "cute"):
                    ModelNew, tmp = load_custom_model_with_tempfile(
                        kernel_code, entry_point="ModelNew"
                    )
                    graceful_eval_cleanup({}, ctx.device, tmp)
                else:
                    ModelNew = load_custom_model(kernel_code, context, build_dir)
                    graceful_eval_cleanup(context, ctx.device)

            if ModelNew is None:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=(
                        "compile_kernel FAILED: ModelNew class not found or "
                        "syntax error prevented execution.\n"
                        f"{stdout_buf.getvalue()}"
                    ),
                    metadata={"compiled": False, "error": "ModelNew not found"},
                )

            return ToolResult(
                tool_name=self.name,
                success=True,
                output="compile_kernel PASSED: kernel compiled without errors.",
                metadata={"compiled": True},
            )

        except Exception as e:
            captured = stdout_buf.getvalue()
            # Keep captured stdout/stderr (that's where nvcc/linker messages land);
            # drop the Python-level traceback — it's frames inside eval.py that
            # the model can't act on.
            detail = captured if captured.strip() else f"{type(e).__name__}: {e}"
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(f"compile_kernel FAILED: {type(e).__name__}.\n{detail}"),
                metadata={"compiled": False, "error": str(e)},
            )


# ---------------------------------------------------------------------------
# 2. RunCorrectnessTool
# ---------------------------------------------------------------------------


class RunCorrectnessTool(Tool):
    name = "run_correctness"
    description = (
        "Run the kernel against the reference for correctness only — no timing. "
        "Use this after compile_kernel succeeds, to verify your kernel produces "
        "numerically equivalent outputs. Returns per-trial pass/fail status and "
        "the nature of any numerical or runtime errors."
    )
    input_schema = _KERNEL_CODE_SCHEMA

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        # Resolve dynamic timeouts (may probe the reference forward once).
        dyn_correctness_s, _dyn_submit_s = _resolve_dynamic_eval_timeouts(ctx)

        # Route through the per-GPU eval server when available so only one
        # run_correctness allocates GPU memory at a time.
        if ctx.eval_client is not None:
            return ctx.eval_client.run_correctness(
                ctx, kernel_code, timeout_s=dyn_correctness_s
            )

        build_dir = _per_kernel_build_dir(ctx.build_dir, kernel_code)

        def _run_correct_once() -> KernelExecResult | None:
            ws = max(1, int(ctx.distributed_torchrun_world_size or 1))
            if ws > 1:
                tr = eval_kernel_via_torchrun(
                    world_size=ws,
                    original_model_src=ctx.ref_arch_src,
                    custom_model_src=kernel_code,
                    seed_num=42,
                    num_correct_trials=ctx.num_correct_trials,
                    num_perf_trials=0,
                    measure_performance=False,
                    timing_method=ctx.timing_method,
                    verbose=ctx.verbose,
                    build_dir=build_dir,
                    backend=ctx.backend,
                    precision_str=ctx.precision,
                    check_for_excessive_speedup=False,
                    timeout_s=int(dyn_correctness_s),
                )
                if tr is not None:
                    return tr
                # Fall through to single-GPU eval.
            return eval_kernel_against_ref(
                original_model_src=ctx.ref_arch_src,
                custom_model_src=kernel_code,
                num_correct_trials=ctx.num_correct_trials,
                num_perf_trials=0,
                measure_performance=False,
                verbose=ctx.verbose,
                build_dir=build_dir,
                device=ctx.device,
                backend=ctx.backend,
                precision=ctx.torch_precision,
                check_for_excessive_speedup=False,
            )

        try:
            result: KernelExecResult | None = _run_with_oom_retry(
                lambda: _retry_eval_on_lock(
                    _run_correct_once, build_dir=build_dir
                )
            )
        except BaseException as exc:
            if _is_cuda_oom(exc):
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=(
                        "run_correctness FAILED: CUDA out of memory after "
                        "retries. The GPU is contested; reduce activation "
                        "memory in your kernel or wait and try again."
                    ),
                    metadata={"error": "cuda_oom"},
                )
            raise

        if result is None:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(
                    "run_correctness FAILED: persistent build/lock contention "
                    "after retries. Please try a different kernel."
                ),
                metadata={},
            )

        if not result.compiled:
            err = result.metadata.get("compilation_error", "unknown error")
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(f"run_correctness FAILED: kernel did not compile.\n{err}"),
                metadata={"compiled": False, "correctness": False},
            )

        trials_str = result.metadata.get("correctness_trials", "?")

        if result.correctness:
            lines = [f"run_correctness PASSED: {trials_str} trials all matched the reference."]
            if result.numerical_precision:
                np_stats = result.numerical_precision
                lines.append(
                    f"Numerical precision: max_abs_err={np_stats.get('max_abs_error', 0):.2e}  "
                    f"mean_abs_err={np_stats.get('mean_abs_error', 0):.2e}  "
                    f"max_rel_err={np_stats.get('max_rel_error', 0):.2e}"
                )
            return ToolResult(
                tool_name=self.name,
                success=True,
                output="\n".join(lines),
                metadata={"compiled": True, "correctness": True, "numerical_precision": result.numerical_precision},
            )

        # Failure path: report the first failing trial and any numeric diffs.
        # We drop `runtime_error_traceback` here — it's frames inside eval.py,
        # not actionable by the model. The core error string lives under
        # `runtime_error` and is surfaced below. Full traceback is still in
        # metadata for human debugging.
        lines = [f"run_correctness FAILED: {trials_str} trials did not all match."]
        for key in ("correctness_issue", "runtime_error"):
            val = result.metadata.get(key)
            if val:
                lines.append(f"{key}: {val}")
        for key in ("max_difference", "avg_difference"):
            val = result.metadata.get(key)
            if val:
                lines.append(f"{key}: {val}")

        return ToolResult(
            tool_name=self.name,
            success=False,
            output="\n".join(lines),
            metadata={"compiled": True, "correctness": False},
        )


# ---------------------------------------------------------------------------
# 3. ProfileKernelTool
# ---------------------------------------------------------------------------


class ProfileKernelTool(Tool):
    name = "profile_kernel"
    description = (
        "Profile the kernel with NVIDIA Nsight Compute. Returns comprehensive "
        "diagnostics: DRAM bandwidth utilization, compute throughput "
        "(FP32/FP16/tensor-core), per-kernel breakdown, warp stall reasons, "
        "memory coalescing quality, shared-memory bank conflicts, L1/L2 hit "
        "rates, occupancy with limiting factors (registers, smem, block size), "
        "pipe utilization, branch divergence, eligible-warps analysis, and "
        "targeted data-driven optimization hints. Supports delta comparison "
        "across iterations. Use when you have a correct kernel and need to "
        "understand why it is slow.\n"
        "Multi-rank mode: when world_size>1 (8xH100 sweeps), profiling runs "
        "ncu --target-processes all around a torchrun --nproc_per_node=N "
        "launch and reports a full ProfileSummary for EACH rank — load "
        "imbalance, divergent stalls, and per-rank occupancy are all "
        "visible. Delta vs previous iteration is computed per rank. "
        "Profiling is fail-hard: any missing rank report or any rank with "
        "zero captured metrics aborts the call."
    )
    input_schema = _KERNEL_CODE_SCHEMA

    _WORKER_SCRIPT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))),
        "scripts", "_profile_worker.py",
    )
    _WORKER_SCRIPT_TORCHRUN = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))),
        "scripts", "_profile_worker_torchrun.py",
    )

    _MULTI_RANK_TIMEOUT_S = 1800

    # Per-tool-instance cache: kernel_src_hash -> ToolResult. Keyed by the
    # exact kernel source the agent passes in, so repeated profile_kernel
    # calls on the same code (common when an agent re-profiles after a
    # non-functional edit) reuse the prior ncu run instead of paying the
    # 30-300s tax again.
    _profile_cache: dict[str, "ToolResult"] = {}

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        if ctx.eval_client is not None:
            return ctx.eval_client.profile_kernel(ctx, kernel_code)

        import hashlib
        import json
        import subprocess
        import sys
        import tempfile

        world_size = max(1, int(getattr(ctx, "distributed_torchrun_world_size", 1) or 1))
        # Include world_size so a single-rank profile and an 8-rank profile
        # of identical code don't collide.
        cache_key = (
            hashlib.sha1(kernel_code.encode("utf-8")).hexdigest() + f":ws{world_size}"
        )
        if cache_key in self._profile_cache:
            cached = self._profile_cache[cache_key]
            return ToolResult(
                tool_name=cached.tool_name,
                success=cached.success,
                output="(cached) " + cached.output,
                metadata=cached.metadata,
            )

        from kernelbench.profile import NSIGHT_AVAILABLE, check_ncu_available
        from kernelbench.agent.nsight_parser import (
            ROOFLINE_METRICS,
            parse_multi_rank_nsight,
            parse_nsight_metrics,
        )

        if not NSIGHT_AVAILABLE:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="profile_kernel FAILED: nsight-python package not installed.",
                metadata={"available": False},
            )
        if not check_ncu_available():
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="profile_kernel FAILED: ncu not found in PATH.",
                metadata={"available": False},
            )

        if world_size > 1:
            return self._execute_multi_rank(
                ctx=ctx,
                kernel_code=kernel_code,
                world_size=world_size,
                cache_key=cache_key,
                parse_multi_rank_nsight=parse_multi_rank_nsight,
                ROOFLINE_METRICS=ROOFLINE_METRICS,
            )

        request = {
            "custom_model_src": kernel_code,
            "ref_model_src": ctx.ref_arch_src,
            "metrics": ROOFLINE_METRICS,
            "num_trials": 1,
            "seed": 42,
            "device_index": ctx.device.index or 0,
            "backend": ctx.backend,
            "precision": ctx.precision,
            "build_dir": _per_kernel_build_dir(ctx.build_dir, kernel_code),
            "verbose": ctx.verbose,
        }

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp:
                json.dump(request, tmp)
                req_path = tmp.name

            proc = subprocess.run(
                [sys.executable, self._WORKER_SCRIPT, req_path],
                capture_output=True,
                text=True,
                timeout=900,
            )

            os.unlink(req_path)

            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "").strip()[-500:]
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=(
                        f"profile_kernel FAILED: worker exited with code "
                        f"{proc.returncode}.\n{stderr_tail}"
                    ),
                    metadata={"error": stderr_tail},
                )

            raw_output = json.loads(proc.stdout.strip().splitlines()[-1])

            if "error" in raw_output:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=f"profile_kernel FAILED: {raw_output['error']}",
                    metadata={"error": raw_output["error"]},
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="profile_kernel FAILED: profiling timed out (900s).",
                metadata={"error": "timeout"},
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"profile_kernel FAILED: {type(e).__name__}: {e}",
                metadata={"error": str(e)},
            )

        kernel_breakdown = raw_output.pop("_kernel_breakdown", [])
        raw_metrics = raw_output

        device_name = torch.cuda.get_device_name(ctx.device)
        previous = ctx._last_profile_summary
        summary = parse_nsight_metrics(
            raw_metrics, device_name, kernel_breakdown=kernel_breakdown
        )

        ctx._last_profile_summary = summary

        result = ToolResult(
            tool_name=self.name,
            success=True,
            output=(
                f"profile_kernel PASSED: profiling complete.\n"
                f"{summary.format_for_llm(previous=previous)}"
            ),
            metadata={
                "raw_metrics": {
                    k: v for k, v in raw_metrics.items() if v is not None
                },
                "bottleneck": summary.bottleneck,
                "dram_utilization_pct": summary.dram_utilization_pct,
                "dominant_pipe": summary.dominant_pipe,
                "dominant_utilization_pct": summary.dominant_utilization_pct,
                "occupancy_pct": summary.occupancy_pct,
                "top_stall": (
                    max(summary.warp_stalls.items(), key=lambda x: x[1])
                    if summary.warp_stalls
                    else None
                ),
            },
        )
        self._profile_cache[cache_key] = result
        return result

    def _execute_multi_rank(
        self,
        *,
        ctx: "ToolContext",
        kernel_code: str,
        world_size: int,
        cache_key: str,
        parse_multi_rank_nsight,
        ROOFLINE_METRICS,
    ) -> "ToolResult":
        """Drive scripts/_profile_worker_torchrun.py and render per-rank summaries.

        Fail-hard semantics: the worker already aborts if any rank is missing
        a report or produced zero metrics. Anything we see here that isn't a
        clean ``{"per_rank": {...}}`` payload is surfaced as PROFILE FAILED.
        """
        import json
        import subprocess
        import sys
        import tempfile

        request = {
            "custom_model_src": kernel_code,
            "ref_model_src": ctx.ref_arch_src,
            "metrics": ROOFLINE_METRICS,
            "num_trials": 1,
            "seed": 42,
            "world_size": world_size,
            "backend": ctx.backend,
            "precision": ctx.precision,
            "build_dir": _per_kernel_build_dir(ctx.build_dir, kernel_code),
            "verbose": ctx.verbose,
            "timeout_s": self._MULTI_RANK_TIMEOUT_S,
        }

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp:
                json.dump(request, tmp)
                req_path = tmp.name

            proc = subprocess.run(
                [sys.executable, self._WORKER_SCRIPT_TORCHRUN, req_path],
                capture_output=True,
                text=True,
                timeout=self._MULTI_RANK_TIMEOUT_S + 60,
            )
            os.unlink(req_path)

            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "").strip()[-800:]
                stdout_tail = (proc.stdout or "").strip()[-800:]
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=(
                        f"profile_kernel FAILED (multi-rank, world_size="
                        f"{world_size}): torchrun worker exited with code "
                        f"{proc.returncode}.\nstdout tail: {stdout_tail}\n"
                        f"stderr tail: {stderr_tail}"
                    ),
                    metadata={
                        "error": stderr_tail or stdout_tail,
                        "world_size": world_size,
                    },
                )

            try:
                raw_output = json.loads(proc.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError) as e:
                stdout_tail = (proc.stdout or "")[-800:]
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=(
                        f"profile_kernel FAILED (multi-rank): could not parse "
                        f"worker JSON: {e}.\nstdout tail: {stdout_tail}"
                    ),
                    metadata={"error": str(e), "world_size": world_size},
                )

            if "error" in raw_output:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=(
                        f"profile_kernel FAILED (multi-rank, world_size="
                        f"{world_size}): {raw_output['error']}"
                    ),
                    metadata={
                        "error": raw_output["error"],
                        "world_size": world_size,
                    },
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(
                    f"profile_kernel FAILED (multi-rank): timed out after "
                    f"{self._MULTI_RANK_TIMEOUT_S}s."
                ),
                metadata={"error": "timeout", "world_size": world_size},
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(
                    f"profile_kernel FAILED (multi-rank): "
                    f"{type(e).__name__}: {e}"
                ),
                metadata={"error": str(e), "world_size": world_size},
            )

        per_rank_raw = raw_output.get("per_rank") or {}
        if not per_rank_raw:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(
                    "profile_kernel FAILED (multi-rank): worker returned no "
                    "per_rank metrics."
                ),
                metadata={"error": "empty per_rank", "world_size": world_size},
            )

        kernel_breakdown = raw_output.get("_kernel_breakdown") or []
        device_name = torch.cuda.get_device_name(ctx.device)

        per_rank_summaries = parse_multi_rank_nsight(
            per_rank_raw=per_rank_raw,
            device_name=device_name,
            kernel_breakdown=kernel_breakdown,
        )

        if len(per_rank_summaries) < world_size:
            missing = sorted(
                set(range(world_size)) - set(per_rank_summaries.keys())
            )
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(
                    f"profile_kernel FAILED (multi-rank): parsed "
                    f"{len(per_rank_summaries)}/{world_size} ranks; "
                    f"missing ranks={missing}."
                ),
                metadata={
                    "error": "incomplete per_rank",
                    "world_size": world_size,
                    "missing_ranks": missing,
                },
            )

        prev_per_rank = ctx._last_per_rank_profile_summary or {}

        rendered: list[str] = [
            f"profile_kernel PASSED: profiling complete "
            f"(world_size={world_size}, all ranks measured separately).\n"
        ]
        for rank_id, summary in per_rank_summaries.items():
            previous = prev_per_rank.get(rank_id)
            rendered.append("")
            rendered.append(f"========== Rank {rank_id} ==========")
            rendered.append(summary.format_for_llm(previous=previous))

        # Persist for next-iteration deltas.
        ctx._last_per_rank_profile_summary = per_rank_summaries

        # Compact metadata: just rank 0 summary plus per-rank gpu_time spread
        gpu_times = [
            s.gpu_time_us for s in per_rank_summaries.values()
            if s.gpu_time_us is not None
        ]
        gpu_time_spread = (
            (max(gpu_times) - min(gpu_times)) if len(gpu_times) >= 2 else None
        )
        rank0 = per_rank_summaries[min(per_rank_summaries.keys())]

        result = ToolResult(
            tool_name=self.name,
            success=True,
            output="\n".join(rendered),
            metadata={
                "world_size": world_size,
                "n_ranks_measured": len(per_rank_summaries),
                "rank0_bottleneck": rank0.bottleneck,
                "rank0_dram_utilization_pct": rank0.dram_utilization_pct,
                "rank0_occupancy_pct": rank0.occupancy_pct,
                "per_rank_gpu_time_us": {
                    str(r): s.gpu_time_us
                    for r, s in per_rank_summaries.items()
                },
                "gpu_time_spread_us": gpu_time_spread,
            },
        )
        self._profile_cache[cache_key] = result
        return result


# ---------------------------------------------------------------------------
# 4. GetGpuSpecsTool (exception to PASS/FAIL rule — reference data only)
# ---------------------------------------------------------------------------


class GetGpuSpecsTool(Tool):
    name = "get_gpu_specs"
    description = (
        "Return peak hardware specs for the GPU this kernel will run on "
        "(memory bandwidth, TFLOPS per precision, SM count, shared memory per "
        "SM, register file size, etc.). Use this once at the start to calibrate "
        "your optimization targets."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    def execute(self, ctx: ToolContext, **_) -> ToolResult:
        if ctx.eval_client is not None:
            return ctx.eval_client.get_gpu_specs(ctx)

        from kernelbench.prompts.hardware.gpu_specs import GPU_SPEC_INFO
        from kernelbench.agent.nsight_parser import _DEVICE_NAME_TO_SPEC_KEY

        device_name = torch.cuda.get_device_name(ctx.device)
        total_mem_gb = (
            torch.cuda.get_device_properties(ctx.device).total_memory / 1024**3
        )

        spec_key = None
        for substr, key in _DEVICE_NAME_TO_SPEC_KEY:
            if substr in device_name:
                spec_key = key
                break

        lines = [
            f"GPU specs for {device_name}:",
            f"  total memory (runtime): {total_mem_gb:.1f} GB",
        ]
        if spec_key and spec_key in GPU_SPEC_INFO:
            lines.append(f"  spec entry: {spec_key}")
            for k, v in GPU_SPEC_INFO[spec_key].items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append("  (no detailed spec entry for this GPU in gpu_specs.py)")

        # Node-shape line. For multi-rank evals (e.g. 8xH100 sweep), reflect
        # the full node so the model knows it can launch on all ranks.
        ws = max(1, int(getattr(ctx, "distributed_torchrun_world_size", 1) or 1))
        if ws > 1:
            try:
                visible = int(torch.cuda.device_count())
            except Exception:
                visible = ws
            lines.append("")
            lines.append(
                f"Node shape: this problem is evaluated on {ws}x {device_name} "
                f"({visible} GPU(s) visible to the parent process). Your "
                f"ModelNew runs inside a torchrun world_size={ws} subprocess; "
                f"each rank owns one device (cuda:LOCAL_RANK) and can use "
                f"torch.distributed collectives (NCCL) across the node."
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            output="\n".join(lines),
            metadata={
                "device_name": device_name,
                "spec_key": spec_key,
                "world_size": ws,
            },
        )


# ---------------------------------------------------------------------------
# 5. StaticCheckTool
# ---------------------------------------------------------------------------


class StaticCheckTool(Tool):
    name = "static_check"
    description = (
        "Run a static-analysis pass that detects reward-hacking patterns "
        "(try/except fallbacks to the reference, timing-function patches, "
        "lazy-tensor tricks, threading injection, etc.). Use this before "
        "submit_kernel as a sanity check — flagged submissions cause "
        "evaluation to fail."
    )
    input_schema = _KERNEL_CODE_SCHEMA

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        valid, errors, warnings = validate_kernel_static(
            code=kernel_code,
            backend=ctx.backend,
            precision=ctx.precision,
        )

        if valid and not warnings:
            output = "static_check PASSED: no violations or warnings detected."
        elif valid:
            lines = ["static_check PASSED (with warnings):"]
            for w in warnings:
                lines.append(f"  WARNING: {w}")
            output = "\n".join(lines)
        else:
            lines = ["static_check FAILED: strict violations found."]
            for e in errors:
                lines.append(f"  ERROR: {e}")
            if warnings:
                lines.append("Advisory warnings:")
                for w in warnings:
                    lines.append(f"  WARNING: {w}")
            output = "\n".join(lines)

        return ToolResult(
            tool_name=self.name,
            success=valid,
            output=output,
            metadata={"valid": valid, "errors": errors, "warnings": warnings},
        )


# ---------------------------------------------------------------------------
# 6. SubmitKernelTool
# ---------------------------------------------------------------------------


class SubmitKernelTool(Tool):
    """
    Final submission: full correctness + timing evaluation.

    Anti-reward-hacking policy:
    - Reports the kernel's absolute runtime in μs.
    - Does NOT reveal the reference runtime or speedup ratio.
    """

    name = "submit_kernel"
    description = (
        "Submit the final kernel for full evaluation: correctness check AND "
        "timing measurement. This ends the session — only call it when you "
        "are confident the kernel is correct and optimized. Returns kernel "
        "runtime in microseconds. The reference runtime and speedup ratio "
        "are NOT revealed."
    )
    input_schema = _KERNEL_CODE_SCHEMA

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        # Resolve dynamic timeouts (may probe the reference forward once).
        _dyn_correctness_s, dyn_submit_s = _resolve_dynamic_eval_timeouts(ctx)

        # If the agent has an eval RPC client wired up (set by run_sweep.py
        # when it spawns per-GPU eval servers), route the heavy correctness
        # + perf-timing work to the server. Decouples eval lifetime from
        # agent worker lifetime so a stuck eval doesn't starve other agents
        # past their worker_timeout_s.
        if ctx.eval_client is not None:
            return ctx.eval_client.submit_kernel(
                ctx, kernel_code, timeout_s=dyn_submit_s
            )

        build_dir = _per_kernel_build_dir(ctx.build_dir, kernel_code)
        try:

            def _run_eval_once() -> KernelExecResult | None:
                ws = max(1, int(ctx.distributed_torchrun_world_size or 1))
                # Multi-rank trigger: world_size > 1 alone. No longer gated on
                # whether the reference imports distributed_collectives — the
                # 8xH100 sweep deliberately runs non-collective oracles on
                # all 8 ranks (rank 0 reports, others redundantly compute).
                if ws > 1:
                    tr = eval_kernel_via_torchrun(
                        world_size=ws,
                        original_model_src=ctx.ref_arch_src,
                        custom_model_src=kernel_code,
                        seed_num=42,
                        num_correct_trials=ctx.num_correct_trials,
                        num_perf_trials=ctx.num_perf_trials,
                        measure_performance=True,
                        timing_method=ctx.timing_method,
                        verbose=ctx.verbose,
                        build_dir=build_dir,
                        backend=ctx.backend,
                        precision_str=ctx.precision,
                        check_for_excessive_speedup=True,
                        timeout_s=int(dyn_submit_s),
                    )
                    if tr is not None:
                        return tr
                    # Too few GPUs visible after clearing CUDA_VISIBLE_DEVICES:
                    # fall through to single-GPU eval.

                return eval_kernel_against_ref(
                    original_model_src=ctx.ref_arch_src,
                    custom_model_src=kernel_code,
                    num_correct_trials=ctx.num_correct_trials,
                    num_perf_trials=ctx.num_perf_trials,
                    measure_performance=True,
                    timing_method=ctx.timing_method,
                    verbose=ctx.verbose,
                    build_dir=build_dir,
                    device=ctx.device,
                    backend=ctx.backend,
                    precision=ctx.torch_precision,
                    check_for_excessive_speedup=True,
                )

            result: KernelExecResult | None = _run_with_oom_retry(
                lambda: _retry_eval_on_lock(
                    lambda: _run_eval_once(),
                    build_dir=build_dir,
                )
            )
        except BaseException as exc:
            if _is_cuda_oom(exc):
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=(
                        "submit_kernel FAILED: CUDA out of memory after "
                        "retries. The GPU is contested or the kernel's "
                        "working set exceeds device memory."
                    ),
                    metadata={"error": "cuda_oom"},
                )
            raise

        if result is None:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(
                    "submit_kernel FAILED: persistent build/lock contention "
                    "after retries. Please try a different kernel."
                ),
                metadata={},
            )

        if not result.compiled:
            err = result.metadata.get("compilation_error", "unknown compilation error")
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(f"submit_kernel FAILED: kernel did not compile.\n{err}"),
                metadata=result.model_dump(),
            )

        if not result.correctness:
            trials_str = result.metadata.get("correctness_trials", "?")
            lines = [
                f"submit_kernel FAILED: correctness check did not pass ({trials_str} trials)."
            ]
            for key in ("correctness_issue", "runtime_error"):
                val = result.metadata.get(key)
                if val:
                    lines.append(f"{key}: {val}")
            for key in ("max_difference", "avg_difference"):
                val = result.metadata.get(key)
                if val:
                    lines.append(f"{key}: {val}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="\n".join(lines),
                metadata=result.model_dump(),
            )

        # Correctness passed — report runtime but NOT speedup.
        trials_str = result.metadata.get("correctness_trials", "?")
        lines = [f"submit_kernel PASSED: {trials_str} correctness trials all passed."]
        if result.runtime > 0:
            lines.append(f"Kernel runtime: {result.runtime:.2f} μs")
            stats = result.runtime_stats
            if stats:

                def _fmt(v: Any) -> str:
                    return f"{v:.2f}" if isinstance(v, (int, float)) else "?"

                lines.append(
                    f"Runtime stats: mean={_fmt(stats.get('mean'))}μs  "
                    f"median={_fmt(stats.get('median'))}μs  "
                    f"std={_fmt(stats.get('std'))}μs"
                )

        # ── Extended metrics summary ──
        if result.numerical_precision:
            np_stats = result.numerical_precision
            lines.append(
                f"Numerical precision: max_abs_err={np_stats.get('max_abs_error', '?'):.2e}  "
                f"mean_abs_err={np_stats.get('mean_abs_error', '?'):.2e}  "
                f"max_rel_err={np_stats.get('max_rel_error', '?'):.2e}"
            )

        if result.memory_stats and result.memory_stats.get("peak_memory_mb"):
            mem = result.memory_stats
            lines.append(
                f"Memory: {mem.get('peak_memory_mb', '?')} MB peak "
                f"(ref: {mem.get('ref_peak_memory_mb', '?')} MB, "
                f"ratio: {mem.get('memory_ratio', '?')}x)"
            )

        if result.kernel_launch_stats and result.kernel_launch_stats.get("num_kernels", -1) > 0:
            kl = result.kernel_launch_stats
            lines.append(
                f"Kernel launches: {kl.get('num_kernels', '?')} "
                f"(ref: {kl.get('ref_num_kernels', '?')}, "
                f"fusion ratio: {kl.get('fusion_ratio', '?')})"
            )

        if result.energy_stats and result.energy_stats.get("energy_per_run_mj", -1) > 0:
            en = result.energy_stats
            lines.append(
                f"Energy: {en.get('energy_per_run_mj', '?')} mJ/run "
                f"(ref: {en.get('ref_energy_per_run_mj', '?')} mJ, "
                f"ratio: {en.get('energy_ratio', '?')}x)"
            )

        if result.sol_stats and result.sol_stats.get("sol_score", -1) >= 0:
            sol = result.sol_stats
            lines.append(f"SOL score: {sol.get('sol_score', '?')}")

        if result.metadata.get("excessive_speedup"):
            lines.append(
                "Flagged for excessive speedup — this submission may have "
                "been rejected by automated review."
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            output="\n".join(lines),
            metadata=result.model_dump(),
        )


# ---------------------------------------------------------------------------
# 7. DisassembleKernelTool
# ---------------------------------------------------------------------------


class DisassembleKernelTool(Tool):
    """
    Disassemble compiled CUDA binary to inspect SASS, PTX, register usage,
    and instruction mix via cuobjdump and nvdisasm.
    """

    name = "disassemble_kernel"
    description = (
        "Disassemble the compiled CUDA kernel to inspect its native GPU "
        "assembly (SASS), PTX intermediate representation, and per-kernel "
        "resource usage (registers, shared memory, spills). Use this when "
        "you have a correct kernel and want to understand the compiler's "
        "code generation — register pressure, instruction mix (memory vs "
        "compute vs control), tensor-core usage, and register spills. "
        "Requires cuobjdump and nvdisasm (shipped with CUDA Toolkit)."
    )
    input_schema = _KERNEL_CODE_SCHEMA

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        if ctx.eval_client is not None:
            return ctx.eval_client.disassemble_kernel(ctx, kernel_code)

        from kernelbench.sass import (
            check_cuobjdump_available,
            check_nvdisasm_available,
            disassemble_kernelbench_model,
        )
        from kernelbench.agent.sass_parser import parse_disassembly

        if not check_cuobjdump_available():
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="disassemble_kernel FAILED: cuobjdump not found in PATH.",
                metadata={"available": False},
            )

        nvdisasm_ok = check_nvdisasm_available()

        try:
            disasm_result = disassemble_kernelbench_model(
                custom_model_src=kernel_code,
                device=ctx.device,
                backend=ctx.backend,
                precision=ctx.torch_precision,
                build_dir=_per_kernel_build_dir(ctx.build_dir, kernel_code),
                include_ptx=True,
                include_nvdisasm=nvdisasm_ok,
                include_life_range=nvdisasm_ok,
                verbose=ctx.verbose,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"disassemble_kernel FAILED: {type(e).__name__}: {e}",
                metadata={"error": str(e)},
            )

        device_name = torch.cuda.get_device_name(ctx.device)
        summary = parse_disassembly(disasm_result, device_name)

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=(
                f"disassemble_kernel PASSED: disassembly analysis complete.\n"
                f"{summary.format_for_llm()}"
            ),
            metadata={
                "max_registers": summary.max_registers,
                "has_register_spills": summary.has_register_spills,
                "has_tensor_core_ops": summary.has_tensor_core_ops,
                "instruction_mix": summary.instruction_mix,
            },
        )


# ---------------------------------------------------------------------------
# 8. ErtRooflineTool
# ---------------------------------------------------------------------------


class ErtRooflineTool(Tool):
    """
    Run empirical roofline micro-benchmarks to measure actual peak bandwidth
    and compute throughput for the current GPU.
    """

    name = "ert_roofline"
    description = (
        "Run the Empirical Roofline Tool — micro-benchmarks that measure "
        "the actual (not theoretical) peak memory bandwidth and compute "
        "throughput of the current GPU. Returns measured bandwidth at each "
        "memory hierarchy level (L1, L2, DRAM/HBM) and peak TFLOPS "
        "(FP32, FP16/tensor-core). Also computes the ridge point — the "
        "arithmetic intensity where the roofline transitions from memory- "
        "to compute-bound. Results are cached per GPU so subsequent calls "
        "are instant. No arguments needed."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    def execute(self, ctx: ToolContext, **_) -> ToolResult:
        if ctx.eval_client is not None:
            return ctx.eval_client.ert_roofline(ctx)

        from kernelbench.ert import run_ert_benchmarks

        try:
            model = run_ert_benchmarks(
                device=ctx.device,
                use_cache=True,
                verbose=ctx.verbose,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"ert_roofline FAILED: {type(e).__name__}: {e}",
                metadata={"error": str(e)},
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=(
                f"ert_roofline PASSED: empirical roofline benchmarks complete.\n"
                f"{model.format_for_llm()}"
            ),
            metadata=model.to_dict(),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Tools that require special hardware/software access and are excluded from
# the default tool set.
_OPT_IN_TOOLS = {"profile_kernel", "disassemble_kernel", "ert_roofline"}

ALL_TOOLS: list[Tool] = [
    CompileKernelTool(),
    RunCorrectnessTool(),
    ProfileKernelTool(),
    GetGpuSpecsTool(),
    StaticCheckTool(),
    SubmitKernelTool(),
    DisassembleKernelTool(),
    ErtRooflineTool(),
]

TOOL_REGISTRY: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}


def get_tools(tool_names: list[str] | None = None) -> list[Tool]:
    """
    Return the list of Tool instances for the given names.

    - tool_names=None → default set (all tools except opt-in tools that
      require special hardware/software: profile_kernel, disassemble_kernel,
      ert_roofline).
    - submit_kernel is always included regardless of the list — without it
      the agent has no way to record a final evaluation result.
    """
    if tool_names is None:
        selected = [t for t in ALL_TOOLS if t.name not in _OPT_IN_TOOLS]
    else:
        wanted = set(tool_names)
        wanted.add("submit_kernel")
        selected = [t for t in ALL_TOOLS if t.name in wanted]
    return selected
