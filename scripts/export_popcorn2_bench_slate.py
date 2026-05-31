#!/usr/bin/env python3
"""Copy finished popcorn2 sweep cells into bench run dirs (15+15+10 slate).

Copies only trajectories that finished in the source run (``finished_at`` set,
``outcome`` not ``in_progress``). Patches ``run_name`` to the bench run. Writes
``sweep_config.json`` from the bench TOML so ``run_sweep`` resume skips copied cells.

Example::

    uv run python scripts/export_popcorn2_bench_slate.py --tier all

Then resume missing work::

    uv run python scripts/run_sweep_eval_stall_watchdog.py configs/sweep.l1234_popcorn2_bench_all.toml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import tomli

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs"
VARIANT = "popcorn2"
SLATE_PATH = REPO_ROOT / "configs" / "popcorn2_bench_slate_v1.toml"

TIER_MAP = {
    "all": {
        "src": "pop_l123_popcorn2_all_gpt",
        "dst": "pop_l123_popcorn2_bench_all_gpt",
        "sweep_toml": REPO_ROOT / "configs" / "sweep.l1234_popcorn2_bench_all.toml",
    },
    # st/default bench sweeps are complete; kept for reference only.
    # "st": {
    #     "src": "pop_l123_popcorn2_st_gpt",
    #     "dst": "pop_l123_popcorn2_bench_st_gpt",
    #     "sweep_toml": REPO_ROOT / "configs" / "sweep.l1234_popcorn2_bench_st.toml",
    # },
    # "default": {
    #     "src": "pop_l123_popcorn2_default_gpt",
    #     "dst": "pop_l123_popcorn2_bench_default_gpt",
    #     "sweep_toml": REPO_ROOT / "configs" / "sweep.l1234_popcorn2_bench_default.toml",
    # },
}

DEFAULT_MODELS = ("gpt-5.5-priority", "FW-GLM-5-1", "grok-4.3")


def load_slate() -> dict[int, list[int]]:
    with open(SLATE_PATH, "rb") as f:
        data = tomli.load(f)
    levels = (data.get("slate") or {}).get("levels")
    if not isinstance(levels, dict):
        raise ValueError(f"could not read [slate.levels] from {SLATE_PATH}")
    return {int(k): list(v) for k, v in levels.items()}


def load_models_from_sweep(toml_path: Path) -> tuple[str, ...]:
    with open(toml_path, "rb") as f:
        sweep = tomli.load(f)
    return tuple(m["name"] for m in sweep.get("models", []))


def trajectory_finished(traj_path: Path) -> bool:
    try:
        with open(traj_path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if not d.get("finished_at"):
        return False
    if d.get("outcome") == "in_progress":
        return False
    return True


def patch_trajectory_run_name(traj_path: Path, run_name: str) -> None:
    with open(traj_path, encoding="utf-8") as f:
        d = json.load(f)
    d["run_name"] = run_name
    d["bench_slate"] = "popcorn2_v2_12_12_10"
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
        f.write("\n")


def prune_bench_run(dst_root: Path, slate: dict[int, list[int]]) -> int:
    """Remove trajectory/kernel files for problem IDs not in the slate."""
    allowed = {
        (level, pid)
        for level, pids in slate.items()
        for pid in pids
    }
    variant_root = dst_root / VARIANT
    if not variant_root.is_dir():
        return 0
    removed = 0
    for traj in variant_root.rglob("*_trajectory.json"):
        stem = traj.stem.replace("_trajectory", "")
        parts = stem.split("_")
        if len(parts) < 4 or parts[0] != "level" or parts[2] != "problem":
            continue
        key = (int(parts[1]), int(parts[3]))
        if key in allowed:
            continue
        for suffix in ("_trajectory.json", "_kernel.py"):
            p = traj.parent / f"{stem}{suffix}"
            if p.is_file():
                p.unlink()
                removed += 1
        cache = traj.parent / f"{stem}_cache"
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)
    return removed


def write_sweep_config_json(dst_run: Path, sweep_toml: Path) -> None:
    with open(sweep_toml, "rb") as f:
        sweep = tomli.load(f)
    dst_run.mkdir(parents=True, exist_ok=True)
    with open(dst_run / "sweep_config.json", "w", encoding="utf-8") as f:
        json.dump(sweep, f, indent=2)
        f.write("\n")


def export_tier(
    *,
    tier: str,
    slate: dict[int, list[int]],
    models: tuple[str, ...],
    dry_run: bool,
    force: bool,
) -> int:
    cfg = TIER_MAP[tier]
    src_root = RUNS / cfg["src"] / VARIANT
    dst_root = RUNS / cfg["dst"]
    dst_variant = dst_root / VARIANT
    run_name = cfg["dst"]

    if not src_root.is_dir():
        print(f"[export] ERROR: missing source {src_root}", file=sys.stderr)
        return 1

    if dst_root.exists() and any(dst_variant.rglob("*_trajectory.json")) and not force:
        print(
            f"[export] ERROR: {dst_root} already has trajectories; use --force to merge",
            file=sys.stderr,
        )
        return 1

    if dry_run:
        print(f"[export] DRY RUN tier={tier} {cfg['src']} -> {cfg['dst']}")
    else:
        dst_variant.mkdir(parents=True, exist_ok=True)
        write_sweep_config_json(dst_root, cfg["sweep_toml"])

    total = 0
    copied = 0
    skipped_in_progress = 0
    missing = 0

    for level, pids in sorted(slate.items()):
        for pid in pids:
            for model in models:
                total += 1
                rel = Path(model) / f"level_{level}_problem_{pid}"
                src_traj = src_root / f"{rel}_trajectory.json"
                src_kernel = src_root / f"{rel}_kernel.py"
                dst_traj = dst_variant / f"{rel}_trajectory.json"
                dst_kernel = dst_variant / f"{rel}_kernel.py"

                if not src_traj.is_file():
                    missing += 1
                    continue
                if not trajectory_finished(src_traj):
                    skipped_in_progress += 1
                    continue

                if dry_run:
                    print(f"  would copy L{level} P{pid} {model} ({src_traj.name})")
                    copied += 1
                    continue

                dst_traj.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_traj, dst_traj)
                patch_trajectory_run_name(dst_traj, run_name)
                if src_kernel.is_file():
                    shutil.copy2(src_kernel, dst_kernel)
                copied += 1

    print(
        f"[export] tier={tier} {cfg['src']} -> {cfg['dst']}: "
        f"copied={copied} skipped_in_progress_or_unfinished={skipped_in_progress} "
        f"missing_src={missing} expected_cells={total}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        choices=("all",),
        default="all",
        help="Which source sweep to export (only all; st/default bench complete).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow exporting into a run dir that already has trajectories.",
    )
    parser.add_argument(
        "--prune-only",
        action="store_true",
        help="Only drop trajectories outside the slate from existing bench dirs.",
    )
    args = parser.parse_args()

    slate = load_slate()
    tiers = [args.tier]
    rc = 0
    if args.prune_only:
        for tier in tiers:
            dst = RUNS / TIER_MAP[tier]["dst"]
            n = prune_bench_run(dst, slate)
            write_sweep_config_json(dst, TIER_MAP[tier]["sweep_toml"])
            print(f"[export] pruned {n} file(s) under {dst.name}")
        return rc
    for tier in tiers:
        models = load_models_from_sweep(TIER_MAP[tier]["sweep_toml"])
        rc |= export_tier(
            tier=tier,
            slate=slate,
            models=models,
            dry_run=args.dry_run,
            force=args.force,
        )
        if not args.dry_run and rc == 0:
            dst = RUNS / TIER_MAP[tier]["dst"]
            n = prune_bench_run(dst, slate)
            if n:
                print(f"[export] pruned {n} file(s) outside slate from {dst.name}")
            write_sweep_config_json(dst, TIER_MAP[tier]["sweep_toml"])
    if not args.dry_run and rc == 0:
        print(
            "[export] Next: build reports and/or resume missing cells:\n"
            "  uv run python scripts/build_report.py runs/pop_l123_popcorn2_bench_all_gpt\n"
            "  uv run python scripts/run_sweep_eval_stall_watchdog.py configs/sweep.l1234_popcorn2_bench_all.toml"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
