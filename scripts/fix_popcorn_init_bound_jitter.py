#!/usr/bin/env python3
"""Revert jitter on dimensions passed to Model.__init__ via get_init_inputs()."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOTS = [
    REPO / "KernelBench" / f"level{n}" / "popcorn"
    for n in (1, 2, 3)
]


def _init_arg_names(src: str) -> set[str]:
    m = re.search(
        r"def get_init_inputs\([^)]*\):\s*\n\s*return\s*\[([^\]]*)\]",
        src,
        re.DOTALL,
    )
    if not m:
        return set()
    raw = m.group(1).strip()
    if not raw:
        return set()
    names: set[str] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            names.append(part.split(".")[0])
    return set(names)


def fix_file(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    if "popcorn_pri" not in src:
        return False
    bound = _init_arg_names(src)
    if not bound:
        return False
    out = src
    for name in bound:
        out = re.sub(
            rf"popcorn_pri\.jitter_int\(\s*{re.escape(name)}\s*(?:,\s*align=\d+)?\s*\)",
            name,
            out,
        )
    if out == src:
        return False
    path.write_text(out, encoding="utf-8")
    return True


def main() -> int:
    n = 0
    for root in ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("[0-9]*.py")):
            if fix_file(path):
                print(f"fixed {path.relative_to(REPO)}")
                n += 1
    print(f"done: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
