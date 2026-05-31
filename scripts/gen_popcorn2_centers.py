#!/usr/bin/env python3
"""
Bootstrap ``KernelBench/level{N}/popcorn2/`` from canonical ``popcorn/``.

Copies each problem and applies ``large``-tier scaling to module-level int
constants (same rules as stress_refs large tier). Per-trial jitter uses
``popcorn_pri.sample_input_mode`` / ``trial_dim`` (see migrate_popcorn2_trial_dims.py).

  uv run python scripts/gen_popcorn2_centers.py
"""

from __future__ import annotations

import ast
import os
import shutil
import sys

REPO_TOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_BENCH = os.path.join(REPO_TOP, "KernelBench")
sys.path.insert(0, os.path.join(REPO_TOP, "src"))

from kernelbench.popcorn_scaling import has_pair_seq2, transform_int  # noqa: E402

LEVELS = (1, 2, 3, 4)
VARIANT_SRC = "popcorn"
VARIANT_DST = "popcorn2"


def _maybe_transform_assign(
    node: ast.Assign, tier: str, pair_attention: bool
) -> ast.Assign:
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return node
    name = node.targets[0].id
    if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, int):
        return node
    new_val = transform_int(name, node.value.value, tier, pair_attention=pair_attention)
    if new_val == node.value.value:
        return node
    return ast.Assign(
        targets=node.targets,
        value=ast.Constant(value=new_val),
        lineno=node.lineno,
        col_offset=node.col_offset,
    )


def _process_file(src_path: str, dst_path: str) -> None:
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    pair = has_pair_seq2(src)
    tree = ast.parse(src)
    new_body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            new_body.append(_maybe_transform_assign(node, "large", pair))
        else:
            new_body.append(node)
    tree.body = new_body
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree)
    header = (
        f"# popcorn2: large-tier module centers (scripts/gen_popcorn2_centers.py).\n"
        f"# Source: {os.path.relpath(src_path, REPO_TOP)}\n\n"
    )
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(header + out)
        if not out.endswith("\n"):
            f.write("\n")


def main() -> int:
    n = 0
    for level in LEVELS:
        src_dir = os.path.join(KERNEL_BENCH, f"level{level}", VARIANT_SRC)
        dst_dir = os.path.join(KERNEL_BENCH, f"level{level}", VARIANT_DST)
        if not os.path.isdir(src_dir):
            print(f"[popcorn2] skip missing {src_dir}", file=sys.stderr)
            continue
        if os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir)
        os.makedirs(dst_dir, exist_ok=True)
        for fn in sorted(os.listdir(src_dir)):
            if not fn.endswith(".py"):
                continue
            src_path = os.path.join(src_dir, fn)
            dst_path = os.path.join(dst_dir, fn)
            _process_file(src_path, dst_path)
            n += 1
            print(f"[popcorn2] {os.path.relpath(dst_path, REPO_TOP)}")
    print(f"[popcorn2] wrote {n} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
