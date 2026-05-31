#!/usr/bin/env python3
"""Migrate popcorn2 get_inputs() to biased trial_dim / sample_input_mode."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys_path = REPO / "src"
import sys

sys.path.insert(0, str(sys_path))

from kernelbench.popcorn_scaling import has_pair_seq2  # noqa: E402

POPCORN2_ROOTS = [
    REPO / "KernelBench" / f"level{n}" / "popcorn2"
    for n in (1, 2, 3, 4)
]

_PAIR_SEQ_NAMES = frozenset({"seq_len", "sequence_length", "max_seq_len", "seq_len_q"})

_MAKE_FUNCS = (
    "make_csr_graph",
    "make_csr_graph_with_edge_weights",
    "make_csr_degree_normalized_graph",
    "make_csr_multihead_spmm_graph",
    "make_csr_scalar_edge_scores",
    "make_csr_multihead_edge_scores",
    "make_csr_fused_attention_value_graph",
    "moe_dispatch_inputs",
    "moe_combine_inputs",
)


def _extract_get_inputs(src: str) -> tuple[str, str, str] | None:
    m = re.search(r"(def get_inputs\(\):\n)((?:    .+\n)+)", src)
    if not m:
        return None
    start, end = m.start(), m.end()
    return src[:start], m.group(1) + m.group(2), src[end:]


def _dim_name(arg: str) -> str:
    arg = arg.strip()
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", arg):
        return arg
    return "_"


def _jitter_to_trial_dim(
    match: re.Match[str], *, pair_attention: bool
) -> str:
    arg = match.group(1).strip()
    rest = match.group(2) or ""
    name = _dim_name(arg)
    extra = f", {rest}" if rest else ""
    pair_kw = ""
    if pair_attention and name in _PAIR_SEQ_NAMES:
        pair_kw = ", pair_attention=True"
    return f"p.trial_dim({arg}, {name!r}, mode=mode{pair_kw}{extra})"


def _add_mode_to_make_calls(block: str) -> str:
    for fn in _MAKE_FUNCS:
        pat = re.compile(
            rf"(\b(?:popcorn_pri|p)\.{fn}\([^)]*)\)(\s*\)?)",
        )
        block = pat.sub(
            lambda m: (
                f"{m.group(1)}, mode=mode){m.group(2)}"
                if "mode=mode" not in m.group(0)
                else m.group(0)
            ),
            block,
        )
    return block


def _ensure_p_and_mode(block: str) -> str:
    lines = block.splitlines()
    if not lines:
        return block
    header = lines[0]
    body = [ln for ln in lines[1:] if ln.strip() != "mode = p.sample_input_mode()"]
    body = [ln.replace("popcorn_pri.", "p.") for ln in body]
    body = [ln for ln in body if not re.match(r"\s+p = popcorn_pri\s*$", ln)]
    if not any("sample_input_mode" in ln for ln in body):
        body = ["    p = popcorn_pri", "    mode = p.sample_input_mode()", *body]
    else:
        if not any(re.match(r"\s+p = popcorn_pri\s*$", ln) for ln in body):
            body = ["    p = popcorn_pri", *body]
    return header + "\n" + "\n".join(body) + ("\n" if block.endswith("\n") else "")


def _apply_pair_attention(block: str) -> str:
    for name in _PAIR_SEQ_NAMES:
        block = re.sub(
            rf"p\.trial_dim\(({re.escape(name)}),\s*{name!r},\s*mode=mode\)",
            rf"p.trial_dim(\1, {name!r}, mode=mode, pair_attention=True)",
            block,
        )
    return block


def _patch_get_inputs_block(block: str, *, pair_attention: bool) -> str:
    block = _ensure_p_and_mode(block)
    block = re.sub(
        r"(?:p|popcorn_pri)\.jitter_int\(([^,)]+)(?:,\s*([^)]*))?\)",
        lambda m: _jitter_to_trial_dim(m, pair_attention=pair_attention),
        block,
    )
    block = _add_mode_to_make_calls(block)
    if pair_attention:
        block = _apply_pair_attention(block)
    return block


def _patch_file(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    if "def get_inputs" not in src:
        return False
    parts = _extract_get_inputs(src)
    if parts is None:
        return False
    prefix, block, suffix = parts
    pair = has_pair_seq2(src)
    new_block = _patch_get_inputs_block(block, pair_attention=pair)
    path.write_text(prefix + new_block + suffix, encoding="utf-8")
    return new_block != block


def main() -> int:
    n = 0
    for root in POPCORN2_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("[0-9]*.py")):
            if _patch_file(path):
                print(f"patched {path.relative_to(REPO)}")
                n += 1
    print(f"done: {n} file(s) updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
