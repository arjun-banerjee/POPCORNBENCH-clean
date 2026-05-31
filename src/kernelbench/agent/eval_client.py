"""Client side of the FIFO eval-queue RPC.

A small helper used by SubmitKernelTool (and optionally ProfileKernelTool)
to push a request onto a per-GPU Manager queue and block on the response.

Each agent worker is given a single response Queue proxy, allocated by the
main process before the worker is spawned. All of that worker's RPC calls
land on that one queue and are demultiplexed by request_id. We can't pass
the Manager itself (not picklable for security reasons), only its proxies.
"""

from __future__ import annotations

import logging
import queue as _queue
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class EvalRPCClient:
    """Thin RPC client owned by an agent process.

    Holds two Manager queue proxies: the shared per-GPU request queue and
    a per-agent response queue. Both are allocated by the main process and
    passed in via run_one's args.
    """

    # Margin between the eval RPC client's response wait and the inner
    # torchrun subprocess timeout. Lets the server detect a torchrun timeout
    # and ship the failure result back to us BEFORE we declare the RPC dead.
    _RPC_MARGIN_S: int = 60

    def __init__(self, request_q, response_q, *, default_timeout_s: int = 3600):
        self._request_q = request_q
        self._response_q = response_q
        self._default_timeout_s = default_timeout_s

    def _ctx_args(self, ctx, *, timeout_s: int | None = None) -> dict:
        # ``timeout_s`` overrides the static ``eval_torchrun_timeout_s`` from
        # the context when a caller (RunCorrectnessTool / SubmitKernelTool)
        # has already resolved a dynamic per-tool timeout. This is what the
        # eval server uses to bound the torchrun subprocess it spawns.
        effective_torchrun_timeout = int(
            timeout_s
            if timeout_s is not None
            else (getattr(ctx, "eval_torchrun_timeout_s", 3600) or 3600)
        )
        return {
            "ref_arch_src": ctx.ref_arch_src,
            "backend": ctx.backend,
            "precision": ctx.precision,
            "build_dir": ctx.build_dir,
            "num_correct_trials": ctx.num_correct_trials,
            "submit_num_correct_trials": int(
                getattr(ctx, "submit_num_correct_trials", ctx.num_correct_trials)
            ),
            "num_perf_trials": ctx.num_perf_trials,
            "timing_method": ctx.timing_method,
            "verbose": ctx.verbose,
            "stream_torchrun_stdout": bool(
                getattr(ctx, "stream_torchrun_stdout", False)
            ),
            # Multi-rank knobs. Servers can ignore these (they're pinned to a
            # single GPU and so cannot drive torchrun), but we forward them
            # for API symmetry — a future "hybrid" eval server could spawn
            # torchrun on demand.
            "distributed_torchrun_world_size": int(
                getattr(ctx, "distributed_torchrun_world_size", 1) or 1
            ),
            "eval_torchrun_timeout_s": effective_torchrun_timeout,
            "level": int(getattr(ctx, "level", 0) or 0),
            "problem_id": int(getattr(ctx, "problem_id", 0) or 0),
            "variant": str(getattr(ctx, "variant", "")),
            "problem_name": str(getattr(ctx, "problem_name", "")),
            "popcorn_stress_eval": bool(getattr(ctx, "popcorn_stress_eval", False)),
            "stress_refs_root": str(
                getattr(ctx, "stress_refs_root", "KernelBench/stress_refs2")
            ),
            "stress_tiers": list(
                getattr(ctx, "stress_tiers", ("large", "awkward", "xl"))
            ),
            "stress_num_correct_trials_per_tier": getattr(
                ctx, "stress_num_correct_trials_per_tier", None
            ),
        }

    def submit_kernel(self, ctx, kernel_code: str, *, timeout_s: int | None = None):
        args = {"kernel_code": kernel_code, **self._ctx_args(ctx, timeout_s=timeout_s)}
        return self._call("submit", args, rpc_timeout_s=self._rpc_wait(timeout_s))

    def compile_kernel(self, ctx, kernel_code: str):
        args = {"kernel_code": kernel_code, **self._ctx_args(ctx)}
        return self._call("compile", args)

    def run_correctness(self, ctx, kernel_code: str, *, timeout_s: int | None = None):
        args = {"kernel_code": kernel_code, **self._ctx_args(ctx, timeout_s=timeout_s)}
        return self._call("correctness", args, rpc_timeout_s=self._rpc_wait(timeout_s))

    def get_gpu_specs(self, ctx):
        args = self._ctx_args(ctx)
        return self._call("specs", args)

    def disassemble_kernel(self, ctx, kernel_code: str):
        args = {"kernel_code": kernel_code, **self._ctx_args(ctx)}
        return self._call("disassemble", args)

    def ert_roofline(self, ctx):
        args = self._ctx_args(ctx)
        return self._call("ert", args)

    def profile_kernel(self, ctx, kernel_code: str):
        args = {
            "kernel_code": kernel_code,
            "previous_summary": ctx._last_profile_summary,
            **self._ctx_args(ctx),
        }
        result, new_summary = self._call("profile", args, return_aux=True)
        # Update the agent's local context so the NEXT profile gets a delta
        ctx._last_profile_summary = new_summary
        return result

    def _rpc_wait(self, inner_timeout_s: int | None) -> int | None:
        """Compute the response-queue wait for a per-call inner timeout.

        ``inner_timeout_s`` is the torchrun subprocess cap the server will
        use. We wait that long plus a small margin so the server has a
        chance to ship back a clean timeout failure before our RPC fires.
        Returns ``None`` to mean "use the default timeout".
        """
        if inner_timeout_s is None:
            return None
        return max(int(inner_timeout_s) + self._RPC_MARGIN_S, self._default_timeout_s)

    def _call(
        self,
        kind: str,
        args: dict,
        *,
        return_aux: bool = False,
        rpc_timeout_s: int | None = None,
    ):
        from kernelbench.agent.tools import ToolResult

        rid = uuid.uuid4().hex

        try:
            self._request_q.put((rid, kind, args, self._response_q))
        except Exception as e:
            logger.error("[eval_client] enqueue failed for %s: %s", kind, e)
            return ToolResult(
                tool_name=f"{kind}_kernel",
                success=False,
                output=f"{kind}_kernel FAILED: enqueue error: {e}",
                metadata={"error": str(e)},
            )

        # Drain the response queue until we see OUR id. Defensive cases:
        #   1. On a sweep restart a stale entry could linger from a prior
        #      worker — different rid, just skip it.
        #   2. If a SIGALRM (work-item soft timeout) fires while we're
        #      blocked inside a Manager-queue get(), the proxy's recv() is
        #      interrupted mid-protocol and the underlying socket is left
        #      in a half-read state. Subsequent get() calls then return
        #      None (or other garbage that won't unpack as a 3-tuple).
        #      We can't fix the corruption from here, but we can prevent
        #      it from poisoning every later RPC for this worker: skip
        #      malformed payloads and keep waiting against an absolute
        #      deadline, so we still time out cleanly.
        wait_s = int(rpc_timeout_s) if rpc_timeout_s is not None else int(self._default_timeout_s)
        end_at = time.monotonic() + wait_s
        while True:
            remaining = end_at - time.monotonic()
            if remaining <= 0:
                return _timeout_result(kind, wait_s, return_aux)
            try:
                payload = self._response_q.get(timeout=remaining)
            except _queue.Empty:
                return _timeout_result(kind, wait_s, return_aux)

            if not isinstance(payload, tuple) or len(payload) != 3:
                logger.warning(
                    "[eval_client] discarding malformed payload (%r) "
                    "while waiting for %s; queue may be poisoned by a "
                    "prior signal-interrupted get()",
                    type(payload).__name__, rid,
                )
                continue

            rid_resp, status, data = payload

            if rid_resp == rid:
                break
            logger.warning("[eval_client] discarding stale response %s "
                           "(waiting for %s)", rid_resp, rid)

        if status == "error":
            err_text = data.get("error", "unknown error")
            tb = data.get("traceback", "")
            result = ToolResult(
                tool_name=f"{kind}_kernel",
                success=False,
                output=f"{kind}_kernel FAILED: {err_text}",
                metadata={"error": err_text, "traceback": tb},
            )
            if return_aux:
                return result, None
            return result

        # Success path: data is the tool's payload dict
        result = ToolResult(
            tool_name=data["tool_name"],
            success=data["success"],
            output=data["output"],
            metadata=data.get("metadata", {}) or {},
        )
        if return_aux:
            return result, data.get("new_profile_summary")
        return result


def _timeout_result(kind: str, timeout_s: int, return_aux: bool):
    from kernelbench.agent.tools import ToolResult
    err = f"{kind}_kernel FAILED: eval server did not respond within {timeout_s}s"
    logger.warning("[eval_client] %s", err)
    result = ToolResult(
        tool_name=f"{kind}_kernel",
        success=False,
        output=err,
        metadata={"error": "eval_rpc_timeout"},
    )
    if return_aux:
        return result, None
    return result
