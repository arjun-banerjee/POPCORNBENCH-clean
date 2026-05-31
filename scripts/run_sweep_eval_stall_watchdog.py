#!/usr/bin/env python3
"""Run ``scripts/run_sweep.py`` and kill the sweep only on true eval stalls.

``run_sweep.py`` prints ``[eval_q]  queue depths: gpu0=...`` about once per
minute. Historically, this watchdog killed the sweep if the **same** depth line
repeated too many times. That caused false positives when one long in-flight
eval kept queue depth unchanged even though work was progressing elsewhere.

Now we require BOTH:
  1) repeated identical eval_q snapshots, and
  2) no progress signals for a configurable grace period.

This keeps sweeps alive while they're still making forward progress.

Exit codes:
  0   — sweep finished successfully
  125 — stale eval_q snapshots (caller may rerun the same TOML)
  else — propagated from ``run_sweep.py`` (caller should not assume stall)
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

EVAL_Q_LINE = re.compile(r"^\[eval_q\]\s+queue depths:")
# Broad "work is advancing" signals from run_sweep output.
PROGRESS_LINE = re.compile(
    r"(turn \d+ LLM done|compile_kernel (OK|FAIL)|run_correctness (OK|FAIL)|"
    r"submit_kernel (PASSED|FAILED)|profile_kernel (PASSED|FAILED)|"
    r"eval server respawned|\[Worker\] crashed|\[run_sweep\] done)",
    re.IGNORECASE,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

EXIT_STALE_EVALQ = 125


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stale-run",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Kill when the same eval_q depth line appears more than N times "
            "in a row (default: 5 → kill on the 6th identical snapshot)."
        ),
    )
    parser.add_argument(
        "--min-stall-minutes",
        type=float,
        default=20.0,
        metavar="M",
        help=(
            "Require at least M minutes without progress before killing on "
            "repeated identical eval_q snapshots (default: 20)."
        ),
    )
    parser.add_argument(
        "config",
        help="Path to sweep TOML (passed to scripts/run_sweep.py).",
    )
    parser.add_argument(
        "run_sweep_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to run_sweep.py after the config path.",
    )
    args = parser.parse_args()
    stale_threshold: int = args.stale_run
    min_stall_seconds = max(0.0, float(args.min_stall_minutes) * 60.0)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_sweep.py"),
        args.config,
        *args.run_sweep_args,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(REPO_ROOT),
        start_new_session=True,
    )

    last_key: str | None = None
    streak = 0
    # Last time we saw concrete progress in sweep output.
    last_progress_ts = time.monotonic()

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

            if PROGRESS_LINE.search(line):
                last_progress_ts = time.monotonic()

            if not EVAL_Q_LINE.match(line):
                continue
            key = line.rstrip("\n")
            if key == last_key:
                streak += 1
            else:
                last_key = key
                streak = 1
            no_progress_for_s = time.monotonic() - last_progress_ts
            if streak > stale_threshold and no_progress_for_s >= min_stall_seconds:
                print(
                    f"[watchdog] same eval_q line {streak} times in a row "
                    f"(>{stale_threshold}) and no progress for "
                    f"{no_progress_for_s/60:.1f} min "
                    f"(>= {min_stall_seconds/60:.1f}); "
                    f"sending SIGTERM to sweep process group",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                sys.exit(EXIT_STALE_EVALQ)
    except KeyboardInterrupt:
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        proc.wait()
        raise

    code = int(proc.wait())
    sys.exit(code)


if __name__ == "__main__":
    main()
