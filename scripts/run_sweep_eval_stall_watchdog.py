#!/usr/bin/env python3
"""Run ``scripts/run_sweep.py`` and kill the sweep if eval queue depth logs stall.

``run_sweep.py`` prints ``[eval_q]  queue depths: gpu0=...`` about once per
minute. If the **same** line repeats more than ``--stale-run`` times in a row,
we assume eval workers are stuck and send SIGTERM to the whole process session.

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
from pathlib import Path

EVAL_Q_LINE = re.compile(r"^\[eval_q\]\s+queue depths:")

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

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if not EVAL_Q_LINE.match(line):
                continue
            key = line.rstrip("\n")
            if key == last_key:
                streak += 1
            else:
                last_key = key
                streak = 1
            if streak > stale_threshold:
                print(
                    f"[watchdog] same eval_q line {streak} times in a row "
                    f"(>{stale_threshold}); sending SIGTERM to sweep process group",
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
