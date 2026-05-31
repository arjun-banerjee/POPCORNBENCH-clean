#!/usr/bin/env python3
"""
Copy a completed hardware_translation sweep run, then serially re-run
submit_kernel-equivalent torchrun eval for every saved ``*_kernel.py``,
writing fresh ``final_result`` + ``outcome`` into the *copied* trajectories only.

Designed for a single 8-GPU node: one eval at a time (no parallelism).

Example:
  uv run python scripts/reval_saved_torchrun_sweep.py \\
    --dst-run-name sl5_hw_translation_8xh100_all_gpt_reval30m \\
    --torchrun-timeout-s 1800

Defaults ``--src-run`` to ``runs/sl5_hw_translation_8xh100_all_gpt`` under the
repo root (the directory that contains ``sweep_config.json``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_TOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_TOP not in sys.path:
    sys.path.insert(0, REPO_TOP)

import torch

from kernelbench.agent.tools import _per_kernel_build_dir
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.distributed_torchrun_eval import eval_kernel_via_torchrun
from kernelbench.eval import KernelExecResult, eval_kernel_against_ref, get_torch_dtype_from_string
from kernelbench.hardware_translation_io import (
    load_io_distributed_world_size,
    load_oracle_reference_source,
)
from kernelbench.utils import set_gpu_arch

_KERNEL_PATH_RE = re.compile(r"level_(\d+)_problem_(\d+)_kernel\.py$")


def _resolve_under_repo(p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(REPO_TOP, p)


def _load_source_kernel(run_cfg: dict[str, Any], problem, level: int) -> str:
    """Match ``run_sweep._load_source_kernel`` (hardware_translation)."""
    src_dir = run_cfg.get("source_kernel_dir")
    if src_dir:
        if not os.path.isabs(src_dir):
            src_dir = os.path.join(REPO_TOP, src_dir)
    else:
        backend = run_cfg.get("source_backend") or run_cfg["backend"]
        src_dir = os.path.join(
            REPO_TOP, "KernelBench", f"level{level}", "_translation_sources", backend
        )
    candidate = os.path.join(src_dir, problem.name)
    if not os.path.exists(candidate):
        stem = os.path.splitext(problem.name)[0]
        for ext in (".cu", ".cuh"):
            alt = os.path.join(src_dir, stem + ext)
            if os.path.exists(alt):
                candidate = alt
                break
        else:
            raise FileNotFoundError(
                f"No source kernel for '{problem.name}' under {src_dir}. "
                "Tried .py, .cu, .cuh extensions."
            )
    with open(candidate, encoding="utf-8") as f:
        return f.read()


def _force_backend_precision(backend: str, precision: str) -> str:
    b = backend.lower()
    if b == "tilelang":
        return "fp16"
    if b == "thunderkittens":
        return "bf16"
    return precision


def _outcome_from_final_result(fr: dict[str, Any] | None) -> str:
    if fr is None:
        return "error"
    if not fr.get("compiled"):
        return "compile_fail"
    if not fr.get("correctness"):
        return "incorrect"
    return "correct"


def _run_submit_like_eval(
    *,
    ref_arch_src: str,
    kernel_code: str,
    world_size: int,
    build_dir_base: str | None,
    run_cfg: dict[str, Any],
    agent_cfg: dict[str, Any],
    torchrun_timeout_s: int,
) -> KernelExecResult:
    """Mirror ``SubmitKernelTool`` local path (no eval RPC)."""
    build_dir = _per_kernel_build_dir(build_dir_base, kernel_code)
    backend = run_cfg["backend"]
    precision = _force_backend_precision(backend, run_cfg["precision"])
    torch_dtype = get_torch_dtype_from_string(precision)
    submit_n = run_cfg.get("submit_num_correct_trials")
    num_correct = int(submit_n) if submit_n is not None else int(run_cfg["num_correct_trials"])
    num_perf = int(run_cfg["num_perf_trials"])
    timing_method = str(run_cfg.get("timing_method", "cuda_event"))
    verbose = bool(agent_cfg.get("verbose", False))
    stream_stdout = bool(agent_cfg.get("stream_torchrun_stdout", False))

    ws = max(1, int(world_size or 1))
    if ws > 1:
        tr = eval_kernel_via_torchrun(
            world_size=ws,
            original_model_src=ref_arch_src,
            custom_model_src=kernel_code,
            seed_num=42,
            num_correct_trials=num_correct,
            num_perf_trials=num_perf,
            measure_performance=True,
            timing_method=timing_method,
            verbose=verbose,
            stream_stdout=stream_stdout,
            build_dir=build_dir,
            backend=backend,
            precision_str=precision,
            check_for_excessive_speedup=True,
            timeout_s=int(torchrun_timeout_s),
        )
        if tr is not None:
            return tr

    return eval_kernel_against_ref(
        original_model_src=ref_arch_src,
        custom_model_src=kernel_code,
        num_correct_trials=num_correct,
        num_perf_trials=num_perf,
        measure_performance=True,
        timing_method=timing_method,
        verbose=verbose,
        build_dir=build_dir,
        device=torch.device("cuda:0"),
        backend=backend,
        precision=torch_dtype,
        check_for_excessive_speedup=True,
    )


def _patch_sweep_config_name(dst_run_dir: Path, new_name: str) -> None:
    cfg_path = dst_run_dir / "sweep_config.json"
    if not cfg_path.is_file():
        print(f"[reval] warning: no {cfg_path}; skip sweep_config patch", flush=True)
        return
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("run", {})["name"] = new_name
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def _stamp_re_eval_metadata(meta: dict[str, Any], *, timeout_s: int, script: str) -> dict[str, Any]:
    out = dict(meta) if meta else {}
    out["re_eval_script"] = script
    out["re_eval_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out["re_eval_torchrun_timeout_s"] = int(timeout_s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src-run",
        default="runs/sl5_hw_translation_8xh100_all_gpt",
        help="Source run directory (relative to repo root unless absolute).",
    )
    ap.add_argument(
        "--dst-run-name",
        required=True,
        help="New directory name under runs/ (must not exist yet). Example: "
        "sl5_hw_translation_8xh100_all_gpt_reval30m",
    )
    ap.add_argument(
        "--torchrun-timeout-s",
        type=int,
        default=1800,
        help="Per-kernel torchrun subprocess cap (default: 1800 = 30 minutes).",
    )
    ap.add_argument(
        "--build-report",
        action="store_true",
        help="Run scripts/build_report.py on the destination run after re-eval.",
    )
    args = ap.parse_args()

    src = Path(_resolve_under_repo(args.src_run))
    if not src.is_dir():
        print(f"[reval] ERROR: source run directory does not exist: {src}", flush=True)
        return 1
    cfg_src = src / "sweep_config.json"
    if not cfg_src.is_file():
        print(f"[reval] ERROR: missing {cfg_src}", flush=True)
        return 1

    dst = Path(REPO_TOP) / "runs" / args.dst_run_name
    if dst.exists():
        print(f"[reval] ERROR: destination already exists (refuse to overwrite): {dst}", flush=True)
        return 1

    print(f"[reval] copytree\n  {src}\n  -> {dst}", flush=True)
    shutil.copytree(src, dst)

    new_run_name = args.dst_run_name
    _patch_sweep_config_name(dst, new_run_name)

    with open(dst / "sweep_config.json", encoding="utf-8") as f:
        sweep_cfg = json.load(f)
    run_cfg = sweep_cfg["run"]
    agent_cfg = sweep_cfg.get("agent", {})

    if run_cfg.get("prompt_option") != "hardware_translation":
        print("[reval] ERROR: sweep is not hardware_translation; aborting.", flush=True)
        shutil.rmtree(dst, ignore_errors=True)
        return 1

    variant = run_cfg["variants"][0]
    level = int(run_cfg["levels"][0])
    gpu_arch = run_cfg.get("gpu_arch", ["Hopper"])
    set_gpu_arch(gpu_arch if isinstance(gpu_arch, list) else [gpu_arch])

    dataset = construct_kernelbench_dataset(
        level=level,
        source=run_cfg["dataset_src"],
        dataset_name=run_cfg.get("dataset_name", "ScalingIntelligence/KernelBench"),
        variant=variant,
    )

    io_dir = run_cfg.get("hardware_translation_io_dir")
    oracle_dir = run_cfg.get("hardware_translation_oracle_dir")
    if not io_dir or not oracle_dir:
        print("[reval] ERROR: run.hardware_translation_io_dir / oracle_dir missing.", flush=True)
        return 1
    if not os.path.isabs(io_dir):
        io_dir = os.path.join(REPO_TOP, io_dir)
    if not os.path.isabs(oracle_dir):
        oracle_dir = os.path.join(REPO_TOP, oracle_dir)

    default_ws = int(agent_cfg.get("distributed_torchrun_world_size", 8) or 8)

    # Parent must not pin a single GPU when driving torchrun world_size>1.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.pop("HIP_VISIBLE_DEVICES", None)

    kernel_files = sorted(dst.glob("**/level_*_problem_*_kernel.py"))
    print(
        f"[reval] found {len(kernel_files)} kernel files; "
        f"torchrun_timeout_s={args.torchrun_timeout_s} (serial)",
        flush=True,
    )

    script_tag = os.path.basename(sys.argv[0])
    for i, kpath in enumerate(kernel_files, 1):
        m = _KERNEL_PATH_RE.search(kpath.name)
        if not m:
            print(f"[reval] skip (unexpected name): {kpath}", flush=True)
            continue
        lev, pid = int(m.group(1)), int(m.group(2))
        traj_path = kpath.with_name(kpath.name.replace("_kernel.py", "_trajectory.json"))
        if not traj_path.is_file():
            print(f"[reval] skip (no trajectory): {kpath}", flush=True)
            continue

        with open(kpath, encoding="utf-8") as f:
            kernel_code = f.read()
        if not kernel_code.strip():
            print(f"[reval] skip (empty kernel): {kpath}", flush=True)
            continue

        print(
            f"[reval] ({i}/{len(kernel_files)}) L{lev}/P{pid} model_dir={kpath.parent.name}",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            problem = dataset.get_problem_by_id(pid)
            ref_arch_src = load_oracle_reference_source(
                repo_top=REPO_TOP,
                oracle_dir=oracle_dir,
                problem_name=problem.name,
            )
            # Touch same paths as run_sweep (validates I/O layout).
            _load_source_kernel(run_cfg, problem, lev)
            ws = load_io_distributed_world_size(
                repo_top=REPO_TOP,
                io_dir=io_dir,
                problem_name=problem.name,
                default=default_ws,
            )
            model_dir = str(kpath.parent)
            cache_base = os.path.join(model_dir, f"level_{lev}_problem_{pid}_cache")
            os.makedirs(cache_base, exist_ok=True)

            result = _run_submit_like_eval(
                ref_arch_src=ref_arch_src,
                kernel_code=kernel_code,
                world_size=ws,
                build_dir_base=cache_base,
                run_cfg=run_cfg,
                agent_cfg=agent_cfg,
                torchrun_timeout_s=args.torchrun_timeout_s,
            )
        except Exception as e:
            print(f"[reval] ERROR eval L{lev}/P{pid}: {e}", flush=True)
            traceback.print_exc()
            result = KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={
                    "compilation_error": f"{type(e).__name__}: {e}",
                    "error": "reval_script_exception",
                },
            )

        fr = result.model_dump(mode="json")
        if isinstance(fr.get("metadata"), dict):
            fr["metadata"] = _stamp_re_eval_metadata(
                fr["metadata"],
                timeout_s=args.torchrun_timeout_s,
                script=script_tag,
            )

        with open(traj_path, encoding="utf-8") as f:
            traj = json.load(f)
        traj["final_result"] = fr
        traj["outcome"] = _outcome_from_final_result(fr)
        traj["run_name"] = new_run_name
        traj["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(traj, f, indent=2)
            f.write("\n")

        dt = time.perf_counter() - t0
        print(
            f"[reval]   -> outcome={traj['outcome']} compiled={fr.get('compiled')} "
            f"correctness={fr.get('correctness')} ({dt:.1f}s wall)",
            flush=True,
        )

    if args.build_report:
        br = os.path.join(REPO_TOP, "scripts", "build_report.py")
        cmd = [sys.executable, br, str(dst)]
        print(f"[reval] running: {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd=REPO_TOP)
        if r.returncode != 0:
            print(f"[reval] build_report exited {r.returncode}", flush=True)
            return r.returncode

    print(f"[reval] done. Destination: {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
