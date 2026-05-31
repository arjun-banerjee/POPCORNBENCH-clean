"""Tests for dynamic eval-timeout plumbing.

We validate three things, all without touching real CUDA / torchrun:

1. ``compute_dynamic_timeout`` matches the documented formula and clamps
   to the configured floor / ceiling.
2. ``_resolve_dynamic_eval_timeouts`` returns the static ceiling when the
   dynamic feature is disabled, and the formula-derived values when it is
   enabled (using a mocked probe).
3. ``RunCorrectnessTool`` / ``SubmitKernelTool`` forward the dynamic
   timeout into ``eval_kernel_via_torchrun(..., timeout_s=...)`` rather
   than the static ceiling.

Run with: uv run pytest src/kernelbench/unit_tests/test_dynamic_eval_timeout.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kernelbench.reference_timing import compute_dynamic_timeout


# ---------------------------------------------------------------------------
# 1. Formula
# ---------------------------------------------------------------------------


def test_compute_dynamic_timeout_returns_ceiling_when_t_ref_none():
    assert (
        compute_dynamic_timeout(
            t_ref_s=None,
            overhead_s=120,
            k=10,
            n_trials=5,
            floor_s=120,
            ceiling_s=5400,
        )
        == 5400
    )


def test_compute_dynamic_timeout_returns_ceiling_when_t_ref_nonpositive():
    assert (
        compute_dynamic_timeout(
            t_ref_s=0.0,
            overhead_s=120,
            k=10,
            n_trials=5,
            floor_s=120,
            ceiling_s=5400,
        )
        == 5400
    )


def test_compute_dynamic_timeout_correctness_formula_at_t_ref_1s():
    # overhead 120 + 10 * 5 * 1.0 = 170; floor=120 -> max(120, 170) = 170;
    # ceiling=5400 -> min(5400, 170) = 170.
    assert (
        compute_dynamic_timeout(
            t_ref_s=1.0,
            overhead_s=120,
            k=10,
            n_trials=5,
            floor_s=120,
            ceiling_s=5400,
        )
        == 170
    )


def test_compute_dynamic_timeout_submit_formula_at_t_ref_1s():
    # overhead 180 + 10 * (5+50) * 1.0 = 730; floor=300 -> max(300, 730) = 730;
    # ceiling=5400 -> 730.
    assert (
        compute_dynamic_timeout(
            t_ref_s=1.0,
            overhead_s=180,
            k=10,
            n_trials=55,  # num_correct_trials + num_perf_trials
            floor_s=300,
            ceiling_s=5400,
        )
        == 730
    )


def test_compute_dynamic_timeout_clamps_to_floor():
    # overhead 120 + 10 * 5 * 0.01 = 120.5 -> below floor 200 -> 200.
    assert (
        compute_dynamic_timeout(
            t_ref_s=0.01,
            overhead_s=120,
            k=10,
            n_trials=5,
            floor_s=200,
            ceiling_s=5400,
        )
        == 200
    )


def test_compute_dynamic_timeout_clamps_to_ceiling():
    # overhead 180 + 10 * 55 * 100 = 55180 -> clamp to ceiling 5400.
    assert (
        compute_dynamic_timeout(
            t_ref_s=100.0,
            overhead_s=180,
            k=10,
            n_trials=55,
            floor_s=300,
            ceiling_s=5400,
        )
        == 5400
    )


# ---------------------------------------------------------------------------
# 2. _resolve_dynamic_eval_timeouts
# ---------------------------------------------------------------------------


class _FakeCtx:
    """Mimics the subset of ToolContext attributes the resolver reads."""

    def __init__(self, **overrides):
        defaults = dict(
            ref_arch_src="class Model: pass",
            backend="cuda",
            precision="fp16",
            device=None,
            build_dir=None,
            verbose=False,
            num_correct_trials=5,
            submit_num_correct_trials=5,
            num_perf_trials=50,
            distributed_torchrun_world_size=8,
            eval_torchrun_timeout_s=5400,
            dynamic_eval_timeout=True,
            reference_probe_enabled=True,
            correctness_overhead_s=120,
            correctness_timeout_k=10.0,
            correctness_floor_s=120,
            submit_overhead_s=180,
            submit_timeout_k=10.0,
            submit_floor_s=300,
            reference_probe_timeout_s=300,
            level=5,
            problem_id=6,
        )
        defaults.update(overrides)
        self.__dict__.update(defaults)
        self._ref_runtime_cache: dict = {}
        self._probe_logged: dict = {}


def test_resolve_returns_ceiling_when_dynamic_disabled():
    from kernelbench.agent.tools import _resolve_dynamic_eval_timeouts

    ctx = _FakeCtx(dynamic_eval_timeout=False)
    dyn_c, dyn_s = _resolve_dynamic_eval_timeouts(ctx)
    assert dyn_c == 5400
    assert dyn_s == 5400


def test_resolve_uses_formula_when_dynamic_enabled():
    from kernelbench.agent import tools as tools_mod

    ctx = _FakeCtx()
    with patch.object(tools_mod, "probe_reference_runtime", return_value=1.0) as probe:
        dyn_c, dyn_s = tools_mod._resolve_dynamic_eval_timeouts(ctx)
    assert probe.called
    # 120 + 10 * 5 * 1.0 = 170; submit: 180 + 10 * (5+50) * 1.0 = 730.
    assert dyn_c == 170
    assert dyn_s == 730


def test_resolve_caches_probe_per_key():
    from kernelbench.agent import tools as tools_mod

    ctx = _FakeCtx()
    with patch.object(tools_mod, "probe_reference_runtime", return_value=2.0) as probe:
        tools_mod._resolve_dynamic_eval_timeouts(ctx)
        tools_mod._resolve_dynamic_eval_timeouts(ctx)
        tools_mod._resolve_dynamic_eval_timeouts(ctx)
    assert probe.call_count == 1


def test_resolve_falls_back_to_ceiling_when_probe_returns_none():
    from kernelbench.agent import tools as tools_mod

    ctx = _FakeCtx()
    with patch.object(tools_mod, "probe_reference_runtime", return_value=None):
        dyn_c, dyn_s = tools_mod._resolve_dynamic_eval_timeouts(ctx)
    assert dyn_c == 5400
    assert dyn_s == 5400


def test_resolve_skips_probe_when_reference_probe_disabled():
    from kernelbench.agent import tools as tools_mod

    ctx = _FakeCtx(reference_probe_enabled=False)
    with patch.object(tools_mod, "probe_reference_runtime", return_value=1.0) as probe:
        dyn_c, dyn_s = tools_mod._resolve_dynamic_eval_timeouts(ctx)
    assert not probe.called
    assert dyn_c == 5400
    assert dyn_s == 5400


def test_resolve_submit_uses_submit_num_correct_trials_distinct_from_run():
    """submit_kernel dyn timeout uses submit_num_correct_trials + num_perf_trials."""
    from kernelbench.agent import tools as tools_mod

    ctx = _FakeCtx(num_correct_trials=2, submit_num_correct_trials=5)
    with patch.object(tools_mod, "probe_reference_runtime", return_value=1.0):
        dyn_c, dyn_s = tools_mod._resolve_dynamic_eval_timeouts(ctx)
    # correctness: 120 + 10 * 2 * 1 = 140
    assert dyn_c == 140
    # submit: 180 + 10 * (5 + 50) * 1 = 730
    assert dyn_s == 730


# ---------------------------------------------------------------------------
# 3. Tools forward the dynamic timeout into eval_kernel_via_torchrun
# ---------------------------------------------------------------------------


def _make_full_ctx(**overrides):
    """Build a real ToolContext (so attribute access matches production)."""
    import torch

    from kernelbench.agent.tools import ToolContext

    defaults = dict(
        ref_arch_src="class Model: pass",
        backend="cuda",
        precision="fp16",
        device=torch.device("cpu"),
        build_dir=None,
        num_correct_trials=5,
        submit_num_correct_trials=5,
        num_perf_trials=50,
        timing_method="cuda_event",
        verbose=False,
        eval_client=None,
        distributed_torchrun_world_size=8,
        eval_torchrun_timeout_s=5400,
        dynamic_eval_timeout=True,
        correctness_overhead_s=120,
        correctness_timeout_k=10.0,
        correctness_floor_s=120,
        submit_overhead_s=180,
        submit_timeout_k=10.0,
        submit_floor_s=300,
        reference_probe_timeout_s=300,
        level=5,
        problem_id=6,
    )
    defaults.update(overrides)
    return ToolContext(**defaults)


def _fake_kernel_exec_result(ok: bool):
    """Minimal stand-in for the KernelExecResult returned by torchrun eval."""
    from kernelbench.eval import KernelExecResult

    return KernelExecResult(
        compiled=True,
        correctness=ok,
        metadata={"correctness_trials": "5/5"},
    )


def test_run_correctness_forwards_dynamic_timeout():
    from kernelbench.agent import tools as tools_mod

    ctx = _make_full_ctx()
    captured: dict = {}

    def fake_eval_torchrun(**kwargs):
        captured.update(kwargs)
        return _fake_kernel_exec_result(ok=True)

    with patch.object(tools_mod, "probe_reference_runtime", return_value=1.0), patch.object(
        tools_mod, "eval_kernel_via_torchrun", side_effect=fake_eval_torchrun
    ):
        tool = tools_mod.RunCorrectnessTool()
        result = tool.execute(ctx, kernel_code="x = 1\n")

    assert captured.get("timeout_s") == 170, captured.get("timeout_s")
    assert captured.get("stream_stdout") is False
    assert captured.get("num_correct_trials") == 5
    assert result.success is True


def test_run_correctness_forwards_stream_torchrun_stdout():
    from kernelbench.agent import tools as tools_mod

    ctx = _make_full_ctx(stream_torchrun_stdout=True)
    captured: dict = {}

    def fake_eval_torchrun(**kwargs):
        captured.update(kwargs)
        return _fake_kernel_exec_result(ok=True)

    with patch.object(tools_mod, "probe_reference_runtime", return_value=1.0), patch.object(
        tools_mod, "eval_kernel_via_torchrun", side_effect=fake_eval_torchrun
    ):
        tools_mod.RunCorrectnessTool().execute(ctx, kernel_code="x = 1\n")

    assert captured.get("stream_stdout") is True


def test_submit_kernel_forwards_dynamic_timeout():
    from kernelbench.agent import tools as tools_mod

    ctx = _make_full_ctx()
    captured: dict = {}

    def fake_eval_torchrun(**kwargs):
        captured.update(kwargs)
        # Provide a result with the perf fields the submit path inspects.
        from kernelbench.eval import KernelExecResult

        return KernelExecResult(
            compiled=True,
            correctness=True,
            runtime=1.234,
            runtime_stats={"mean": 1.234},
            metadata={"correctness_trials": "5/5"},
        )

    with patch.object(tools_mod, "probe_reference_runtime", return_value=1.0), patch.object(
        tools_mod, "eval_kernel_via_torchrun", side_effect=fake_eval_torchrun
    ):
        tool = tools_mod.SubmitKernelTool()
        tool.execute(ctx, kernel_code="x = 1\n")

    assert captured.get("timeout_s") == 730, captured.get("timeout_s")
    assert captured.get("num_correct_trials") == 5


def test_submit_kernel_forwards_submit_num_correct_trials_distinct_from_run():
    from kernelbench.agent import tools as tools_mod

    ctx = _make_full_ctx(num_correct_trials=2, submit_num_correct_trials=7)
    captured: dict = {}

    def fake_eval_torchrun(**kwargs):
        captured.update(kwargs)
        from kernelbench.eval import KernelExecResult

        return KernelExecResult(
            compiled=True,
            correctness=True,
            runtime=1.234,
            runtime_stats={"mean": 1.234},
            metadata={"correctness_trials": "7/7"},
        )

    with patch.object(tools_mod, "probe_reference_runtime", return_value=1.0), patch.object(
        tools_mod, "eval_kernel_via_torchrun", side_effect=fake_eval_torchrun
    ):
        tools_mod.SubmitKernelTool().execute(ctx, kernel_code="x = 1\n")

    assert captured.get("num_correct_trials") == 7


def test_eval_rpc_client_run_correctness_forwards_timeout():
    """The eval RPC path replaces eval_torchrun_timeout_s per-call AND
    extends the response-queue wait."""
    from kernelbench.agent.eval_client import EvalRPCClient

    request_q = MagicMock()
    response_q = MagicMock()
    client = EvalRPCClient(request_q, response_q, default_timeout_s=3600)

    ctx = _make_full_ctx()

    captured_call: dict = {}

    def fake_call(kind, args, *, return_aux=False, rpc_timeout_s=None):
        captured_call["kind"] = kind
        captured_call["args"] = args
        captured_call["rpc_timeout_s"] = rpc_timeout_s
        from kernelbench.agent.tools import ToolResult

        return ToolResult(tool_name=f"{kind}_kernel", success=True, output="ok")

    with patch.object(client, "_call", side_effect=fake_call):
        client.run_correctness(ctx, "k=1", timeout_s=170)

    assert captured_call["kind"] == "correctness"
    assert captured_call["args"]["eval_torchrun_timeout_s"] == 170
    assert captured_call["args"].get("stream_torchrun_stdout") is False
    assert captured_call["args"]["num_correct_trials"] == 5
    assert captured_call["args"]["submit_num_correct_trials"] == 5
    # RPC wait = max(170 + 60, default 3600) = 3600 here (default dominates).
    assert captured_call["rpc_timeout_s"] == 3600


def test_eval_rpc_client_rpc_wait_uses_inner_plus_margin_when_larger():
    from kernelbench.agent.eval_client import EvalRPCClient

    request_q = MagicMock()
    response_q = MagicMock()
    client = EvalRPCClient(request_q, response_q, default_timeout_s=600)

    ctx = _make_full_ctx()
    captured_call: dict = {}

    def fake_call(kind, args, *, return_aux=False, rpc_timeout_s=None):
        captured_call["rpc_timeout_s"] = rpc_timeout_s
        from kernelbench.agent.tools import ToolResult

        return ToolResult(tool_name=f"{kind}_kernel", success=True, output="ok")

    with patch.object(client, "_call", side_effect=fake_call):
        client.submit_kernel(ctx, "k=1", timeout_s=1800)

    # max(1800 + 60, 600) = 1860.
    assert captured_call["rpc_timeout_s"] == 1860
