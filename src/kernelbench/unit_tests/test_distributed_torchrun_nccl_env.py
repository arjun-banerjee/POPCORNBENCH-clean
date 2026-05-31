"""Tests for torchrun child env NCCL fail-fast defaults."""

from __future__ import annotations

from kernelbench.distributed_torchrun_eval import (
    apply_torchrun_child_nccl_failfast_defaults,
)


def test_apply_sets_defaults_on_empty_env():
    env: dict = {}
    apply_torchrun_child_nccl_failfast_defaults(env)
    assert env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "1"
    assert env["NCCL_ASYNC_ERROR_HANDLING"] == "1"


def test_apply_does_not_override_existing_values():
    env = {
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0",
        "NCCL_ASYNC_ERROR_HANDLING": "0",
    }
    apply_torchrun_child_nccl_failfast_defaults(env)
    assert env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "0"
    assert env["NCCL_ASYNC_ERROR_HANDLING"] == "0"


def test_torchrun_shutdown_grace_env(monkeypatch):
    import kernelbench.distributed_torchrun_eval as m

    monkeypatch.delenv("KERNELBENCH_TORCHRUN_SHUTDOWN_GRACE_S", raising=False)
    monkeypatch.setattr(m, "_DEFAULT_TORCHRUN_SHUTDOWN_GRACE_S", 60.0)
    assert m._torchrun_shutdown_grace_s() == 60.0

    monkeypatch.setenv("KERNELBENCH_TORCHRUN_SHUTDOWN_GRACE_S", "90")
    assert m._torchrun_shutdown_grace_s() == 90.0

    monkeypatch.setenv("KERNELBENCH_TORCHRUN_SHUTDOWN_GRACE_S", "2")
    assert m._torchrun_shutdown_grace_s() == 5.0  # floored at 5s
