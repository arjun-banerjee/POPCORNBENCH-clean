"""
Multi-rank (torchrun + ncu) profiling worker for the 8xH100 hardware-translation
sweep.

This is a sibling to ``scripts/_profile_worker.py``. Unlike the single-rank
worker — which is driven by ``nsight-python``'s ``@nsight.analyze.kernel``
decorator (it re-execs the script under ncu) — the torchrun path must drive
ncu manually. The decorator does not compose with ``torchrun --nproc_per_node=N``:
each rank would re-invoke ncu, giving N nested ncu invocations.

Operating modes
---------------
Parent mode (default invocation):
    python scripts/_profile_worker_torchrun.py <request.json>

    Builds:
        ncu --target-processes all --csv --page raw
            --metrics <metrics, ...>
            --export <tmpdir>/rep_%h_%p
            -- torchrun --nproc_per_node=N
                       <self> --mode rank <request.json> <tmpdir>

    Then maps PID -> rank via the per-rank pid files each rank writes, parses
    each rank's ncu CSV, and emits ``{"per_rank": {rank_id: {metric: value, ...},
    ...}, "_kernel_breakdown": [...]}`` on stdout.

Rank mode (re-invoked by torchrun for each rank):
    python scripts/_profile_worker_torchrun.py --mode rank <request.json> <tmpdir>

    Each rank:
      1. set ``torch.cuda.set_device(LOCAL_RANK)``
      2. ``dist.init_process_group("nccl")``
      3. load + instantiate ``ModelNew`` on its device
      4. run a single forward pass (this is what ncu attaches to)
      5. write ``<tmpdir>/rank_<LOCAL_RANK>.pid`` containing the current PID
      6. ``dist.barrier()``
      7. exit

Fail-hard contract
------------------
The parent exits 1 with ``{"error": "..."}`` if:
  - torchrun returns a non-zero exit code,
  - any rank's pid file is missing after the run,
  - any rank's ncu CSV produces zero metric values (or fails to parse),
  - the temp dir contains fewer ncu reports than ranks.

Output format
-------------
Successful response on stdout (last JSON line):
    {
      "per_rank": {0: {"dram__bytes_read.sum": 1234, ...}, 1: {...}, ...},
      "_kernel_breakdown": [<torch.profiler breakdown from rank 0>]
    }

Failure response on stdout (last JSON line) + exit 1:
    {"error": "<description>"}
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any


# ---------------------------------------------------------------------------
# Rank mode: invoked by torchrun, one process per device
# ---------------------------------------------------------------------------


def _rank_main(request_path: str, tmpdir: str) -> int:
    """Run a single torchrun rank: load model, forward once, drop pid file."""
    import torch
    import torch.distributed as dist

    with open(request_path) as f:
        req = json.load(f)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        try:
            dist.init_process_group(backend="nccl")
        except Exception as e:
            # Couldn't init NCCL — still write pid file with an error marker so
            # the parent's pid-presence check passes (we want the parent to
            # report a useful error rather than hang).
            with open(os.path.join(tmpdir, f"rank_{local_rank}.pid"), "w") as f:
                f.write(f"{os.getpid()}\nerror: init_process_group: {e}\n")
            return 1

    custom_model_src = req["custom_model_src"]
    ref_model_src = req.get("ref_model_src") or custom_model_src
    backend = req.get("backend", "cuda")
    precision = req.get("precision", "fp32")
    build_dir = req.get("build_dir")
    seed = int(req.get("seed", 42))

    from kernelbench.eval import (
        _process_input_tensor,
        get_torch_dtype_from_string,
        graceful_eval_cleanup,
        load_custom_model,
        load_custom_model_with_tempfile,
        load_original_model_and_inputs,
        set_seed,
    )

    torch_precision = get_torch_dtype_from_string(precision)
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    context: dict = {}
    tempfile_handle = None

    try:
        _, get_init_inputs, get_inputs = load_original_model_and_inputs(
            ref_model_src, context
        )

        set_seed(seed)
        init_inputs = [
            _process_input_tensor(x, device, backend, torch_precision)
            for x in get_init_inputs()
        ]

        if backend.lower() in ("triton", "tilelang", "cute"):
            ModelNew, tempfile_handle = load_custom_model_with_tempfile(
                custom_model_src, entry_point="ModelNew"
            )
        else:
            ModelNew = load_custom_model(custom_model_src, {}, build_dir)
        torch.cuda.synchronize(device=device)

        with torch.no_grad():
            set_seed(seed)
            custom_model = ModelNew(*init_inputs)
            custom_model = custom_model.to(device=device, dtype=torch_precision)
            torch.cuda.synchronize(device=device)

        set_seed(seed)
        inputs = [
            _process_input_tensor(x, device, backend, torch_precision)
            for x in get_inputs()
        ]

        # Rank-0 torch.profiler breakdown (the others discard their output).
        # We pass it back to the parent by writing a JSON next to the pid file.
        if rank == 0:
            try:
                import torch.autograd.profiler as _profiler

                with torch.no_grad():
                    with _profiler.profile(use_cuda=True) as prof:
                        custom_model(*inputs)
                    torch.cuda.synchronize(device=device)

                breakdown = []
                for e in prof.function_events:
                    if e.cuda_time_total > 0 and not e.key.startswith("cudaDevice"):
                        breakdown.append(
                            {
                                "name": e.key,
                                "cuda_time_us": e.cuda_time_total,
                                "calls": e.count,
                            }
                        )
                breakdown.sort(key=lambda x: x["cuda_time_us"], reverse=True)
                with open(
                    os.path.join(tmpdir, "kernel_breakdown.json"), "w"
                ) as f:
                    json.dump(breakdown, f)
            except Exception:
                pass

        # THIS is the ncu-instrumented forward — keep it as the single
        # measurable region. Everything before this was setup.
        with torch.no_grad():
            custom_model(*inputs)
        torch.cuda.synchronize(device=device)

        # Write our PID file AFTER the measured forward so the parent can map
        # PID -> rank using ncu's ``rep_%h_%p`` output.
        with open(os.path.join(tmpdir, f"rank_{local_rank}.pid"), "w") as f:
            f.write(f"{os.getpid()}\n")

        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
        return 0
    except Exception as e:
        # Drop a pid file with an error marker so the parent gives a useful
        # diagnostic instead of "missing pid file".
        try:
            with open(
                os.path.join(tmpdir, f"rank_{local_rank}.pid"), "w"
            ) as f:
                f.write(f"{os.getpid()}\nerror: {type(e).__name__}: {e}\n")
        except Exception:
            pass
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
        return 1
    finally:
        try:
            graceful_eval_cleanup(context, device, tempfile_handle)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Parent mode: drives ncu around torchrun
# ---------------------------------------------------------------------------


_NCU_REPORT_RE = re.compile(r"rep_[^_]+_(\d+)\.ncu-rep$")


def _parent_main(request_path: str) -> int:
    """Drive ncu + torchrun, parse outputs, emit per-rank metrics."""
    with open(request_path) as f:
        req = json.load(f)

    world_size = int(req.get("world_size", 1))
    if world_size < 2:
        print(
            json.dumps(
                {
                    "error": (
                        "world_size must be >= 2 for the torchrun profile "
                        f"path (got {world_size}); use _profile_worker.py "
                        "for single-rank."
                    )
                }
            )
        )
        return 1

    metrics = req.get("metrics") or []
    if not metrics:
        print(json.dumps({"error": "request has no `metrics` list"}))
        return 1

    timeout_s = int(req.get("timeout_s", 1800))
    if subprocess.run(["which", "ncu"], capture_output=True).returncode != 0:
        print(json.dumps({"error": "ncu not found in PATH"}))
        return 1

    with tempfile.TemporaryDirectory(prefix="kb_ncu_torchrun_") as tmpdir:
        report_prefix = os.path.join(tmpdir, "rep_%h_%p")
        metrics_str = ",".join(metrics)

        # The ncu --csv/--page raw output also goes to stdout; we still need
        # the .ncu-rep files (one per rank) for per-PID parsing. ncu writes
        # one file per measured process when --target-processes all is set.
        ncu_argv = [
            "ncu",
            "--target-processes",
            "all",
            "--metrics",
            metrics_str,
            "--csv",
            "--page",
            "raw",
            "--export",
            report_prefix,
            "--force-overwrite",
            "--",
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={world_size}",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=localhost:0",
            "--no-python",
            sys.executable,
            os.path.abspath(__file__),
            "--mode",
            "rank",
            request_path,
            tmpdir,
        ]

        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("HIP_VISIBLE_DEVICES", None)
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env.setdefault("MASTER_PORT", "0")
        env.setdefault("NCCL_DEBUG", "WARN")

        t0 = time.time()
        try:
            proc = subprocess.run(
                ncu_argv,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            print(
                json.dumps(
                    {
                        "error": (
                            f"ncu+torchrun timed out after {timeout_s}s "
                            f"(world_size={world_size})"
                        )
                    }
                )
            )
            return 1
        ncu_wall = time.time() - t0

        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "")[-2000:]
            stdout_tail = (proc.stdout or "")[-1000:]
            print(
                json.dumps(
                    {
                        "error": (
                            f"ncu+torchrun exited {proc.returncode} after "
                            f"{ncu_wall:.1f}s. stderr tail: {stderr_tail!r}; "
                            f"stdout tail: {stdout_tail!r}"
                        )
                    }
                )
            )
            return 1

        # 1) Per-rank pid files must all be present (fail-hard).
        pid_to_rank: dict[int, int] = {}
        missing_ranks: list[int] = []
        rank_errors: list[str] = []
        for r in range(world_size):
            p = os.path.join(tmpdir, f"rank_{r}.pid")
            if not os.path.isfile(p):
                missing_ranks.append(r)
                continue
            try:
                lines = open(p).read().splitlines()
                if not lines:
                    missing_ranks.append(r)
                    continue
                pid = int(lines[0].strip())
                pid_to_rank[pid] = r
                if len(lines) > 1 and lines[1].startswith("error:"):
                    rank_errors.append(f"rank {r}: {lines[1]}")
            except Exception as e:
                rank_errors.append(f"rank {r}: pid parse failed: {e}")
                missing_ranks.append(r)

        if missing_ranks or rank_errors:
            print(
                json.dumps(
                    {
                        "error": (
                            f"fail-hard: missing pid files for ranks "
                            f"{missing_ranks}; rank errors: {rank_errors}"
                        )
                    }
                )
            )
            return 1

        # 2) Map ncu reports to ranks via their pid encoded in the filename
        # (we exported with --export rep_%h_%p, so the suffix is the pid).
        reports = sorted(glob.glob(os.path.join(tmpdir, "rep_*.ncu-rep")))
        report_for_rank: dict[int, str] = {}
        for path in reports:
            m = _NCU_REPORT_RE.search(os.path.basename(path))
            if not m:
                continue
            try:
                pid = int(m.group(1))
            except ValueError:
                continue
            r = pid_to_rank.get(pid)
            if r is not None:
                report_for_rank[r] = path

        # ncu can also have produced extra reports for ephemeral spawn
        # children (e.g. elastic_launch). If we're missing ranks, try a
        # generous fallback that pairs reports in sorted order to ranks
        # sorted by their pid. But if the explicit map already covered all
        # ranks, prefer that.
        if len(report_for_rank) < world_size:
            ranks_sorted = sorted(pid_to_rank.values())
            paths_sorted = list(reports)
            if len(paths_sorted) >= world_size:
                report_for_rank = dict(
                    zip(ranks_sorted, paths_sorted[: world_size])
                )

        missing_reports = [r for r in range(world_size) if r not in report_for_rank]
        if missing_reports:
            print(
                json.dumps(
                    {
                        "error": (
                            f"fail-hard: ncu produced no report for ranks "
                            f"{missing_reports} (found {len(reports)} "
                            f"reports for world_size={world_size})"
                        )
                    }
                )
            )
            return 1

        # 3) Parse each per-rank report via `ncu --import` to CSV. The CSV
        # has one section per measured kernel; we collapse to a single
        # metric->value dict by summing across kernels (matching the
        # single-rank worker's combine_kernel_metrics=sum behaviour).
        per_rank: dict[int, dict[str, float]] = {}
        for r in range(world_size):
            try:
                per_rank[r] = _parse_ncu_report(report_for_rank[r], metrics)
            except Exception as e:
                print(
                    json.dumps(
                        {
                            "error": (
                                f"fail-hard: rank {r} CSV parse failed: "
                                f"{type(e).__name__}: {e}"
                            )
                        }
                    )
                )
                return 1

            # Fail-hard if a rank produced zero usable metrics — that means
            # ncu attached but didn't capture anything (kernel never ran on
            # that rank, or the measured forward was a no-op).
            non_null = sum(1 for v in per_rank[r].values() if v is not None)
            if non_null == 0:
                print(
                    json.dumps(
                        {
                            "error": (
                                f"fail-hard: rank {r} produced zero ncu "
                                f"metrics (kernel never ran on this rank?)"
                            )
                        }
                    )
                )
                return 1

        # 4) Rank-0 kernel breakdown (best-effort; not part of fail-hard).
        breakdown_path = os.path.join(tmpdir, "kernel_breakdown.json")
        kernel_breakdown: list = []
        if os.path.isfile(breakdown_path):
            try:
                with open(breakdown_path) as f:
                    kernel_breakdown = json.load(f) or []
            except Exception:
                kernel_breakdown = []

        # Output: single JSON line on stdout.
        result: dict[str, Any] = {
            "per_rank": {str(k): v for k, v in per_rank.items()},
            "_kernel_breakdown": kernel_breakdown,
            "world_size": world_size,
            "wall_clock_s": ncu_wall,
        }
        print(json.dumps(result))
        return 0


def _parse_ncu_report(report_path: str, metrics: list[str]) -> dict[str, float | None]:
    """Run ``ncu --import <report> --csv --page raw`` and collapse to one dict.

    ncu emits CSV with one row per (kernel, metric) tuple. We sum across
    kernels for each requested metric, matching the single-rank worker's
    ``combine_kernel_metrics=sum`` behaviour.
    """
    argv = [
        "ncu",
        "--import",
        report_path,
        "--csv",
        "--page",
        "raw",
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ncu --import failed (rc={proc.returncode}): "
            f"{(proc.stderr or '')[-500:]}"
        )

    out: dict[str, float | None] = {m: None for m in metrics}
    text = proc.stdout or ""
    # The CSV starts after a banner of "==PROF==" lines. Find the first
    # comma-separated line containing the column header.
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("==")]
    if not lines:
        return out
    # ncu raw CSV columns include "Metric Name" / "Metric Value".
    rdr = csv.DictReader(lines)
    name_col = None
    value_col = None
    for cand in rdr.fieldnames or []:
        c = cand.strip()
        cl = c.lower()
        if cl == "metric name":
            name_col = c
        elif cl == "metric value":
            value_col = c
    if name_col is None or value_col is None:
        # Some ncu versions emit "Metric Name", others variants. Try a fuzzy
        # match.
        for cand in rdr.fieldnames or []:
            cl = cand.lower()
            if name_col is None and "metric" in cl and "name" in cl:
                name_col = cand
            if value_col is None and "value" in cl:
                value_col = cand
    if name_col is None or value_col is None:
        raise RuntimeError(
            f"ncu CSV had no Metric Name / Metric Value columns; "
            f"fields={rdr.fieldnames}"
        )

    for row in rdr:
        name = (row.get(name_col) or "").strip()
        val_s = (row.get(value_col) or "").strip()
        if not name or not val_s:
            continue
        try:
            v = float(val_s.replace(",", ""))
        except ValueError:
            continue
        if name in out:
            prev = out[name]
            out[name] = (prev or 0.0) + v
    return out


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--mode",
        choices=("parent", "rank"),
        default="parent",
        help="invocation mode; 'rank' is used by torchrun children",
    )
    parser.add_argument("request_path", help="path to the request JSON")
    parser.add_argument(
        "tmpdir",
        nargs="?",
        default=None,
        help="rank mode only: shared tmpdir for pid/breakdown files",
    )
    args = parser.parse_args()

    if args.mode == "rank":
        if not args.tmpdir:
            print(json.dumps({"error": "rank mode requires tmpdir argument"}))
            return 1
        return _rank_main(args.request_path, args.tmpdir)
    return _parent_main(args.request_path)


if __name__ == "__main__":
    sys.exit(main())
