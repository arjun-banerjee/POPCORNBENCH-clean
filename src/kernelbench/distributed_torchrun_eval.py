"""Multi-rank (torchrun) evaluation harness for KernelBench problems.

The single-GPU code path (``kernelbench.eval.eval_kernel_against_ref``) loads
the reference ``Model`` and the candidate ``ModelNew`` on one device and
compares outputs. That breaks for problems whose forward pass uses NCCL
collectives, real tensor parallelism, or pipeline parallelism: those models
expect every participating rank to be inside their forward at the same time.

This module wraps the single-rank eval in a ``torchrun`` subprocess so all
ranks load the same models, run the same inputs, and synchronize through
NCCL whenever the model code calls ``torch.distributed``. The eval result
(correctness + timing) is gathered on rank 0 and returned to the caller as a
:class:`kernelbench.eval.KernelExecResult`.

On every rank, ``dist.destroy_process_group()`` is invoked in a ``finally``
block after the eval path (including barriers) so TCPStore / NCCL teardown is
less racy when the subprocess exits.

Design choices
--------------
1. **No gating on "is the reference collective?"**  The plan explicitly drops
   the ``reference_uses_torchrun_collectives`` gate so non-collective oracles
   can also run on an 8-rank node. For them, all ranks redundantly compute the
   same forward — rank 0's result is returned, the others' are discarded.
   Wall-clock cost is identical to single-rank for non-collective models (they
   run in parallel on independent GPUs).
2. **Subprocess isolation.**  We launch ``torchrun`` as a subprocess rather
   than ``mp.spawn`` because the parent (a sweep worker or eval server)
   typically has CUDA already initialized; ``torchrun`` gives us a clean
   environment and clears ``CUDA_VISIBLE_DEVICES`` so all 8 GPUs are visible
   to the worker ranks.
3. **Rank 0 produces the result, others barrier.**  Avoids cross-rank
   aggregation logic in the common case. For perf timing, rank 0's number is
   reported.
4. **Fail-soft (returns None) when too few GPUs are visible.**  The caller
   (``SubmitKernelTool.execute``) falls back to single-GPU eval in that case.
5. **NCCL fail-fast defaults on the torchrun child env.**  We set
   ``TORCH_NCCL_ASYNC_ERROR_HANDLING`` / ``NCCL_ASYNC_ERROR_HANDLING`` (unless
   already set) so a rank error tends to abort NCCL peers instead of wedging
   every rank until ``timeout_s``. This does not remove application-level
   deadlocks or all driver hang classes; the subprocess timeout remains the
   hard backstop.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import timedelta
from typing import Optional

import torch


def apply_torchrun_child_nccl_failfast_defaults(env: dict) -> None:
    """Set default env vars on a torchrun child ``env`` (mutates in place).

    Uses ``setdefault`` so users can override. Intended for every subprocess
    that launches ``torch.distributed.run`` for multi-GPU KernelBench evals.
    """
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")


# ---------------------------------------------------------------------------
# Heuristic: does the reference source import the distributed_collectives shim?
# ---------------------------------------------------------------------------
#
# Kept as a public function for diagnostics / older callers. The
# ``SubmitKernelTool`` / ``RunCorrectnessTool`` paths no longer gate on this:
# any problem whose ToolContext.distributed_torchrun_world_size > 1 takes the
# torchrun branch (see scripts/run_sweep.py + tools.py).
_DC_NEEDLES = (
    "kernelbench.distributed_collectives",
    "from kernelbench.distributed_collectives",
    "import kernelbench.distributed_collectives",
)


def reference_uses_torchrun_collectives(ref_src: str) -> bool:
    """Return True iff the reference source imports the distributed_collectives shim.

    Historically gated the torchrun branch in submit_kernel / run_correctness.
    The current behaviour (per the 8xH100 plan) is: any problem with
    world_size > 1 runs under torchrun regardless of this flag. The function
    is still useful as a diagnostic in trajectories.
    """
    if not ref_src:
        return False
    return any(n in ref_src for n in _DC_NEEDLES)


# ---------------------------------------------------------------------------
# Subprocess worker (this same module is re-invoked by torchrun)
# ---------------------------------------------------------------------------


def _destroy_process_group_if_initialized() -> None:
    """Best-effort teardown so TCPStore / NCCL heartbeat threads shut down cleanly."""
    import torch.distributed as dist

    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _worker_main(request_path: str, result_path: str) -> None:
    """Inner rank entrypoint: load models, run eval, rank-0 writes the result.

    Invoked by every torchrun-spawned rank as ``python -m
    kernelbench.distributed_torchrun_eval --worker <req> <res>``. We rebuild
    the same call shape as the single-GPU ``eval_kernel_against_ref`` and run
    it on the rank's local device. NCCL is initialized so any
    ``torch.distributed`` call inside the model code can complete.

    Two modes are supported via the ``mode`` key in the request payload:
      - default / "eval": full correctness + perf eval against a candidate
        kernel (the historical path).
      - "ref_only": time the reference forward only (no custom kernel).
        Used by ``kernelbench.reference_timing.probe_reference_runtime`` to
        derive dynamic timeouts for run_correctness / submit_kernel.
    """
    import torch.distributed as dist

    from kernelbench.eval import (
        KernelExecResult,
        eval_kernel_against_ref,
        get_torch_dtype_from_string,
    )

    with open(request_path) as f:
        req = json.load(f)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        try:
            import inspect

            init_kw: dict = {
                "backend": backend,
                "timeout": timedelta(minutes=30),
            }
            # PyTorch 2.4+: bind this rank's GPU explicitly (quiets "Guessing
            # device ID" and avoids some heterogeneous-mapping hangs).
            if backend == "nccl" and torch.cuda.is_available():
                if "device_id" in inspect.signature(dist.init_process_group).parameters:
                    init_kw["device_id"] = device
            dist.init_process_group(**init_kw)
        except Exception as e:
            # Init can fail when only 1 GPU is actually visible (e.g. parent
            # pinned CUDA_VISIBLE_DEVICES). Surface as a clean error.
            if rank == 0:
                with open(result_path, "w") as f:
                    json.dump(
                        {
                            "error": (
                                "init_process_group failed: "
                                f"{type(e).__name__}: {e} "
                                f"(world_size={world_size}, visible_gpus="
                                f"{torch.cuda.device_count()})"
                            ),
                        },
                        f,
                    )
            return

    try:
        mode = req.get("mode", "eval")
        if mode == "ref_only":
            _ref_only_worker(req, result_path, device, rank)
            if dist.is_initialized():
                try:
                    dist.barrier()
                except Exception:
                    pass
            return

        try:
            result = eval_kernel_against_ref(
                original_model_src=req["original_model_src"],
                custom_model_src=req["custom_model_src"],
                seed_num=int(req.get("seed_num", 42)),
                num_correct_trials=int(req.get("num_correct_trials", 5)),
                num_perf_trials=int(req.get("num_perf_trials", 0)),
                measure_performance=bool(req.get("measure_performance", False)),
                timing_method=req.get("timing_method", "cuda_event"),
                verbose=bool(req.get("verbose", False)),
                build_dir=req.get("build_dir"),
                device=device,
                backend=req.get("backend", "cuda"),
                precision=get_torch_dtype_from_string(req.get("precision_str", "fp32")),
                check_for_excessive_speedup=bool(
                    req.get("check_for_excessive_speedup", False)
                ),
            )
        except Exception as e:
            if rank == 0:
                with open(result_path, "w") as f:
                    json.dump(
                        {
                            "error": (
                                f"eval_kernel_against_ref raised: "
                                f"{type(e).__name__}: {e}"
                            ),
                        },
                        f,
                    )
            if dist.is_initialized():
                try:
                    dist.barrier()
                except Exception:
                    pass
            return

        # Rank 0 writes the canonical result; everyone else barriers and exits.
        if rank == 0:
            if result is None:
                payload = {"error": "eval returned None (lock contention)"}
            else:
                payload = {"result": json.loads(result.model_dump_json())}
            with open(result_path, "w") as f:
                json.dump(payload, f)

        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
    finally:
        _destroy_process_group_if_initialized()


def _ref_only_worker(req: dict, result_path: str, device, rank: int) -> None:
    """Time the reference forward only; rank 0 writes ``mean_s`` to ``result_path``.

    Used by ``kernelbench.reference_timing.probe_reference_runtime`` to
    derive dynamic per-tool timeouts. We mirror the model-loading shape of
    ``eval_kernel_against_ref`` so the probe pays the same NCCL init +
    forward-pass cost as the operational eval — but never touches a
    candidate kernel.
    """
    from kernelbench.eval import (
        load_original_model_and_inputs,
        set_seed,
        get_torch_dtype_from_string,
    )

    seed_num = int(req.get("seed_num", 42))
    num_trials = max(1, int(req.get("num_trials", 5)))
    precision_dtype = get_torch_dtype_from_string(req.get("precision_str", "fp32"))

    try:
        context: dict = {}
        loaded = load_original_model_and_inputs(req["ref_arch_src"], context)
        if loaded is None:
            raise RuntimeError("load_original_model_and_inputs returned None")
        Model, get_init_inputs, get_inputs = loaded
        if Model is None or get_init_inputs is None or get_inputs is None:
            raise RuntimeError("reference source missing Model/get_inputs")

        def _to_dev(t):
            if not isinstance(t, torch.Tensor):
                return t
            if not t.is_floating_point():
                return t.to(device=device)
            return t.to(device=device, dtype=precision_dtype)

        set_seed(seed_num)
        init_inputs = [_to_dev(x) for x in get_init_inputs()]
        set_seed(seed_num)
        inputs = [_to_dev(x) for x in get_inputs()]

        with torch.no_grad():
            model = Model(*init_inputs).to(device=device, dtype=precision_dtype)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device=device)

        with torch.no_grad():
            _ = model(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=device)

        elapsed: list[float] = []
        for _ in range(num_trials):
            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.no_grad():
                    _ = model(*inputs)
                end.record()
                torch.cuda.synchronize(device=device)
                elapsed.append(start.elapsed_time(end) / 1000.0)
            else:
                import time as _t
                t0 = _t.perf_counter()
                with torch.no_grad():
                    _ = model(*inputs)
                elapsed.append(_t.perf_counter() - t0)

        if rank == 0:
            mean_s = float(sum(elapsed) / len(elapsed)) if elapsed else 0.0
            with open(result_path, "w") as f:
                json.dump(
                    {"mean_s": mean_s, "trials": elapsed, "world_size": int(os.environ.get("WORLD_SIZE", "1"))},
                    f,
                )
    except Exception as e:
        if rank == 0:
            with open(result_path, "w") as f:
                json.dump(
                    {
                        "error": (
                            f"ref_only_worker raised: {type(e).__name__}: {e}"
                        )
                    },
                    f,
                )


# ---------------------------------------------------------------------------
# Parent-side launcher
# ---------------------------------------------------------------------------

_DEFAULT_TORCHRUN_SHUTDOWN_GRACE_S = 60.0


def _torchrun_shutdown_grace_s() -> float:
    """Seconds to wait after SIGTERM before SIGKILL (env override)."""
    raw = os.environ.get("KERNELBENCH_TORCHRUN_SHUTDOWN_GRACE_S", "")
    if not str(raw).strip():
        return _DEFAULT_TORCHRUN_SHUTDOWN_GRACE_S
    try:
        return max(5.0, float(raw))
    except (TypeError, ValueError):
        return _DEFAULT_TORCHRUN_SHUTDOWN_GRACE_S


def _terminate_torchrun_process_tree(proc: subprocess.Popen, *, grace_s: float) -> None:
    """SIGTERM the whole process group, wait ``grace_s``, then SIGKILL if needed.

    Used when ``eval_kernel_via_torchrun`` hits its wall clock so ranks can
    run ``destroy_process_group()`` / NCCL teardown instead of an immediate
    hard kill from ``subprocess.run(..., timeout=...)``.
    """
    import signal
    import time

    if proc.poll() is not None:
        return
    pgid: int | None = None
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, AttributeError, OSError):
        pgid = None
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + max(5.0, float(grace_s))
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if proc.poll() is not None:
        return
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
    else:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        pass


def eval_kernel_via_torchrun(
    *,
    world_size: int,
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 5,
    num_perf_trials: int = 0,
    measure_performance: bool = True,
    timing_method: str = "cuda_event",
    verbose: bool = False,
    stream_stdout: bool = False,
    build_dir: Optional[str] = None,
    backend: str = "cuda",
    precision_str: str = "fp32",
    check_for_excessive_speedup: bool = False,
    timeout_s: int = 3600,
):
    """Spawn ``torchrun --nproc_per_node=N`` and run an 8-rank eval.

    Returns
    -------
    Optional[KernelExecResult]
        ``None`` when too few GPUs are visible after clearing
        ``CUDA_VISIBLE_DEVICES`` (caller falls back to single-GPU eval) or
        when the subprocess returns no parseable result. Otherwise a
        deserialized ``KernelExecResult`` from rank 0.

    Notes
    -----
    The subprocess inherits the parent's environment EXCEPT ``CUDA_VISIBLE_DEVICES``,
    which is cleared so all physical GPUs become visible to torchrun (each
    rank then pins to ``cuda:LOCAL_RANK``). This only works when the parent
    worker is itself running with ``multi_gpu=true`` (no per-worker CVD pin);
    otherwise we'd be lying about device availability to NCCL.

    When ``stream_stdout`` is True, worker stdout/stderr are not captured and
    inherit the parent's terminal (live logs); multi-rank runs may interleave.

    **Timeouts:** the parent uses ``Popen`` + ``wait(timeout=...)`` instead of
    ``subprocess.run(timeout=...)`` so that on expiry we **SIGTERM the whole
    process group**, wait ``KERNELBENCH_TORCHRUN_SHUTDOWN_GRACE_S`` seconds
    (default 60), then **SIGKILL** if still alive. That gives worker ranks a
    chance to tear down NCCL / TCPStore more cleanly than an immediate hard
    kill (still not guaranteed for every hang).
    """
    from kernelbench.eval import KernelExecResult

    if world_size < 2:
        # No work for the torchrun branch — caller should use the single-GPU
        # path. Returning None signals fall-through.
        return None

    # Quick sanity: do we have enough physical GPUs to satisfy the world?
    if torch.cuda.is_available():
        physical = _count_physical_gpus()
        if physical < world_size:
            if verbose:
                print(
                    f"[torchrun_eval] only {physical} physical GPU(s) "
                    f"visible, want {world_size}; falling back to single-GPU."
                )
            return None

    request = {
        "original_model_src": original_model_src,
        "custom_model_src": custom_model_src,
        "seed_num": seed_num,
        "num_correct_trials": num_correct_trials,
        "num_perf_trials": num_perf_trials,
        "measure_performance": measure_performance,
        "timing_method": timing_method,
        "verbose": verbose,
        "build_dir": build_dir,
        "backend": backend,
        "precision_str": precision_str,
        "check_for_excessive_speedup": check_for_excessive_speedup,
    }

    with tempfile.TemporaryDirectory(prefix="kb_torchrun_eval_") as tmp:
        req_path = os.path.join(tmp, "request.json")
        res_path = os.path.join(tmp, "result.json")
        with open(req_path, "w") as f:
            json.dump(request, f)

        env = os.environ.copy()
        # Make ALL physical GPUs visible to torchrun. The parent sweep worker
        # is running multi_gpu=true so this is safe (no single-GPU pin to
        # respect). If a parent ever calls us with CVD set to one GPU, this
        # clears the pin for the subprocess only.
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("HIP_VISIBLE_DEVICES", None)
        apply_torchrun_child_nccl_failfast_defaults(env)
        # PyTorch's distributed init_method "env://" reads these:
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env.setdefault("MASTER_PORT", "0")  # let torchrun pick
        # Quieter NCCL by default; users can override with their own NCCL_*.
        env.setdefault("NCCL_DEBUG", "WARN")
        if stream_stdout:
            # Helps trial-level prints from worker ranks flush promptly.
            env["PYTHONUNBUFFERED"] = "1"

        argv = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={world_size}",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=localhost:0",
            "--no-python",
            # We re-invoke this module's __main__ in worker mode below.
            sys.executable,
            "-m",
            "kernelbench.distributed_torchrun_eval",
            "--worker",
            req_path,
            res_path,
        ]

        grace_s = _torchrun_shutdown_grace_s()
        popen_kw: dict = {
            "env": env,
            "start_new_session": True,
        }
        if stream_stdout:
            popen_kw["stdout"] = None
            popen_kw["stderr"] = subprocess.STDOUT
        else:
            popen_kw["stdout"] = subprocess.PIPE
            popen_kw["stderr"] = subprocess.PIPE
            popen_kw["text"] = True

        proc = subprocess.Popen(argv, **popen_kw)
        timed_out = False
        try:
            proc.wait(timeout=int(timeout_s))
        except subprocess.TimeoutExpired:
            timed_out = True
            if verbose:
                print(
                    f"[torchrun_eval] timeout after {timeout_s}s; "
                    f"SIGTERM process group, grace {grace_s:.0f}s, then SIGKILL if needed.",
                    flush=True,
                )
            _terminate_torchrun_process_tree(proc, grace_s=grace_s)

        rc = proc.poll()
        if rc is None:
            rc = proc.wait()

        captured_out = ""
        captured_err = ""
        if not stream_stdout and proc.stdout is not None:
            captured_out = proc.stdout.read() or ""
        if not stream_stdout and proc.stderr is not None:
            captured_err = proc.stderr.read() or ""

        if timed_out:
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    "compilation_error": (
                        f"torchrun eval timed out after {timeout_s}s "
                        f"(world_size={world_size}); sent SIGTERM then SIGKILL "
                        f"after {grace_s:.0f}s grace"
                    ),
                    "error": "torchrun_timeout",
                    "timeout_shutdown_grace_s": grace_s,
                },
            )

        if not os.path.exists(res_path):
            if stream_stdout:
                tail_out = "(stdout/stderr were streamed; not captured)"
                tail_err = ""
            else:
                tail_out = captured_out[-1000:]
                tail_err = captured_err[-1000:]
            if verbose:
                print(
                    f"[torchrun_eval] no result file produced; rc="
                    f"{rc}\nstdout tail:\n{tail_out}\n"
                    f"stderr tail:\n{tail_err}"
                )
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    "compilation_error": (
                        f"torchrun eval produced no result file "
                        f"(rc={rc}). stderr tail: {tail_err[-400:]}"
                    ),
                    "error": "torchrun_no_result",
                },
            )

        with open(res_path) as f:
            payload = json.load(f)

        if "error" in payload:
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    "compilation_error": payload["error"],
                    "error": "torchrun_worker_error",
                },
            )

        try:
            return KernelExecResult.model_validate(payload["result"])
        except Exception as e:
            return KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    "compilation_error": (
                        f"could not parse torchrun result: {e}"
                    ),
                    "error": "torchrun_parse_error",
                },
            )


def _count_physical_gpus() -> int:
    """Return the number of physical GPUs the parent process can see.

    Unlike ``torch.cuda.device_count()``, this consults nvidia-smi when
    available so we don't have to rely on CUDA having been initialized yet
    (it usually has been by the time this is called, but we want to be
    defensive). Falls back to torch's count on failure.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return sum(1 for l in out.stdout.splitlines() if l.strip())
    except Exception:
        pass
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# CLI: torchrun re-invokes this module with --worker to run a single rank
# ---------------------------------------------------------------------------


def _cli() -> None:
    """torchrun --no-python entrypoint: parse args, run the rank worker."""
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2], sys.argv[3])
        return
    raise SystemExit(
        "kernelbench.distributed_torchrun_eval is a library + worker. "
        "Use eval_kernel_via_torchrun() from Python."
    )


if __name__ == "__main__":
    _cli()
