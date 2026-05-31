#!/usr/bin/env python3
"""Migrator: popcorn problems use per-trial randomized inputs (anti reward-hack)."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POPCORN_ROOTS = [
    REPO / "KernelBench" / f"level{n}" / "popcorn"
    for n in (1, 2, 3, 4)
]

SKIP_JITTER_NAMES = frozenset(
    {
        "vocab_size",
        "k",
        "topk",
        "world_size",
        "rank",
        "device",
        "dtype",
    }
)

# Module ints that must match get_init_inputs / nn constructors (do not jitter).
SKIP_IF_IN_INIT = True

CSR_SCALAR_EDGE = """def get_inputs():
    return list(popcorn_pri.make_csr_scalar_edge_scores(num_nodes, avg_degree))
"""

CSR_MULTIHEAD_EDGE = """def get_inputs():
    return list(popcorn_pri.make_csr_multihead_edge_scores(num_nodes, avg_degree, num_heads))
"""

CSR_SPmm = """def get_inputs():
    return list(popcorn_pri.make_csr_graph_with_edge_weights(num_nodes, avg_degree, feat_dim))
"""

CSR_DEGREE_NORM = """def get_inputs():
    return list(popcorn_pri.make_csr_degree_normalized_graph(num_nodes, avg_degree, feat_dim))
"""

CSR_MULTIHEAD_SPMM = """def get_inputs():
    return list(
        popcorn_pri.make_csr_multihead_spmm_graph(
            num_nodes, avg_degree, num_heads, head_dim
        )
    )
"""

CSR_FUSED_ATTN = """def get_inputs():
    return list(
        popcorn_pri.make_csr_fused_attention_value_graph(
            num_nodes, avg_degree, feat_dim
        )
    )
"""

MOE_DISPATCH = """def get_inputs():
    return popcorn_pri.moe_dispatch_inputs(num_tokens, hidden_dim, num_experts)
"""

MOE_COMBINE = """def get_inputs():
    return popcorn_pri.moe_combine_inputs(num_tokens, hidden_dim, fanout)
"""

MOE_ROUTING = """def get_inputs():
    p = popcorn_pri
    nt = p.jitter_int(num_tokens)
    hd = p.jitter_int(hidden_dim, align=16)
    ne = p.jitter_int(num_experts, minimum=2)
    token_hidden = torch.randn(nt, hd, dtype=torch.float32)
    router_logits = torch.randn(nt, ne, dtype=torch.float32)
    expert_ground = torch.randn(ne, hd, dtype=torch.float32)
    return [token_hidden, router_logits, expert_ground, alpha]
"""

COO_SCATTER = """def get_inputs():
    p = popcorn_pri
    n_edges = p.jitter_int(num_edges)
    fd = p.jitter_int(feat_dim, align=8)
    dst_idx = torch.randint(0, num_nodes, (n_edges,), dtype=torch.int32)
    edge_feat = torch.randn(n_edges, fd, dtype=torch.float32)
    return [dst_idx, edge_feat]
"""

SAMPLED_MATMUL = """def get_inputs():
    p = popcorn_pri
    n_edges = p.jitter_int(num_edges)
    fd = p.jitter_int(feat_dim, align=8)
    ns = p.jitter_int(num_src_nodes)
    nd = p.jitter_int(num_dst_nodes)
    src_idx = torch.randint(0, ns, (n_edges,), dtype=torch.int32)
    dst_idx = torch.randint(0, nd, (n_edges,), dtype=torch.int32)
    src_feat = torch.randn(ns, fd, dtype=torch.float32)
    dst_feat = torch.randn(nd, fd, dtype=torch.float32)
    return [src_idx, dst_idx, src_feat, dst_feat]
"""

L4_GET_INPUTS = """def get_inputs():
    p = popcorn_pri
    bs = p.jitter_int(batch_size)
    sl = p.jitter_int(sequence_length)
    inputs = torch.randint(0, vocab_size, (bs, sl))
    return [inputs]
"""


def _replace_get_inputs_block(src: str, new_body: str) -> str:
    pat = re.compile(r"def get_inputs\(\):\n(?:    .+\n)+", re.MULTILINE)
    if not pat.search(src):
        return src
    return pat.sub(new_body + "\n", src, count=1)


def _strip_make_helpers(src: str) -> str:
    for name in ("_make_row_ptr", "_make_graph"):
        pat = re.compile(
            rf"def {name}\([^)]*\):.*?(?=\n\ndef |\nclass |\Z)",
            re.DOTALL,
        )
        src = pat.sub("", src)
    return src


def _strip_degree_jitter_lines(src: str) -> str:
    lines = [
        ln
        for ln in src.splitlines()
        if "torch.arange(num_nodes" not in ln
        and "degree_jitter" not in ln
        and "degree = torch.clamp(degree +" not in ln
        and "degree = torch.full((num_nodes" not in ln
        and "row_ptr[1:] = torch.cumsum(degree" not in ln
        and "row_ptr = torch.zeros(num_nodes + 1" not in ln
        and "expert_idx = torch.arange(num_tokens" not in ln
        and "token_idx = torch.arange(num_tokens" not in ln
    ]
    return "\n".join(lines) + ("\n" if src.endswith("\n") else "")


def _init_bound_names(src: str) -> set[str]:
    m = re.search(
        r"def get_init_inputs\([^)]*\):\s*\n\s*return\s*\[([^\]]*)\]",
        src,
        re.DOTALL,
    )
    if not m or not m.group(1).strip():
        return set()
    names: set[str] = set()
    for part in m.group(1).split(","):
        part = part.strip()
        if not part or part in ("None",):
            continue
        names.add(part.split(".")[0])
    return names


def _module_int_constants(src: str) -> set[str]:
    names: set[str] = set()
    for m in re.finditer(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\d+)\s*$", src, re.M):
        names.add(m.group(1))
    return names


def _extract_get_inputs_body(src: str) -> tuple[str, str, str] | None:
    m = re.search(r"(def get_inputs\(\):\n)((?:    .+\n)+)", src)
    if not m:
        return None
    start, end = m.start(), m.end()
    return src[:start], m.group(1) + m.group(2), src[end:]


def _generic_patch_get_inputs(src: str) -> str:
    """Jitter module-level int shape constants inside ``get_inputs`` only."""
    if "popcorn_pri" in src:
        return src
    parts = _extract_get_inputs_body(src)
    if parts is None:
        return src
    prefix, block, suffix = parts
    if "popcorn_pri" in block:
        return src

    init_bound = _init_bound_names(src)
    consts = _module_int_constants(src)
    jitter_names = sorted(
        (consts - init_bound - SKIP_JITTER_NAMES),
        key=len,
        reverse=True,
    )

    lines = block.splitlines()
    out_lines: list[str] = []
    inserted_p = False
    for ln in lines:
        if ln.strip() == "def get_inputs():":
            out_lines.append(ln)
            continue
        if not inserted_p and ln.startswith("    ") and ln.strip():
            out_lines.append("    p = popcorn_pri")
            inserted_p = True
        out_lines.append(ln)
    if not inserted_p:
        out_lines.insert(1, "    p = popcorn_pri")
    body = "\n".join(out_lines) + "\n"

    for name in jitter_names:
        if f"p.jitter_int({name}" in body:
            continue
        align = ", align=8" if name.endswith(("_dim", "dim", "hidden", "channels")) else ""
        body = re.sub(
            rf"\b{re.escape(name)}\b",
            f"p.jitter_int({name}{align})",
            body,
        )

    def _jitter_literal(m: re.Match[str]) -> str:
        n = int(m.group(1))
        if n <= 4:
            return m.group(0)
        return f"p.jitter_int({n})"

    body = re.sub(r"(?<=[(,])\s*(\d{2,})\s*(?=[,)])", lambda m: f" p.jitter_int({m.group(1)}) ", body)
    body = re.sub(
        r"torch\.randn\(\s*(\d{2,})\s*,",
        lambda m: f"torch.randn(p.jitter_int({m.group(1)}),",
        body,
    )
    body = re.sub(
        r"torch\.randint\([^,]+,\s*\(\s*(\d{2,})\s*,",
        lambda m: f"torch.randint(0, vocab_size, (p.jitter_int({m.group(1)}),"
        if "vocab_size" in body
        else m.group(0),
        body,
        count=1,
    )

    return prefix + body + suffix


def _jitter_name(name: str, bound: set[str]) -> str:
    if name in bound:
        return name
    if name in ("batch_size", "bs", "B", "R"):
        return f"p.jitter_int({name})"
    if name in ("seq_len", "sl", "length", "time_steps", "S_MAX", "sequence_length", "seq_len_q", "seq_len_kv", "msa_depth"):
        return f"p.jitter_int({name})"
    if name in ("num_heads", "num_atoms", "num_points", "num_train", "num_test", "num_edges"):
        return f"p.jitter_int({name})"
    if name in ("num_tokens", "num_nodes", "num_src_nodes", "num_dst_nodes", "num_experts"):
        return f"p.jitter_int({name})"
    if name in ("D", "feat_dim", "hidden_dim", "hidden", "pair_dim", "msa_dim"):
        return f"p.jitter_int({name}, align=8)"
    if name in ("height", "width", "seq_m", "seq_n"):
        return f"p.jitter_int({name})"
    return name


def _jitter_conv_get_inputs(src: str) -> str:
    bound = _init_bound_names(src)
    pat = re.compile(
        r"def get_inputs\(\):\n"
        r"    return \[torch\.randn\(([^)]+)\)\]\n",
        re.MULTILINE,
    )

    def repl(m: re.Match[str]) -> str:
        parts = [p.strip() for p in m.group(1).split(",")]
        jittered = [_jitter_name(p, bound) for p in parts]
        return (
            "def get_inputs():\n"
            f"    p = popcorn_pri\n"
            f"    return [torch.randn({', '.join(jittered)})]\n"
        )

    return pat.sub(repl, src)


def _jitter_dual_seq_attention(src: str) -> str:
    pat = re.compile(
        r"def get_inputs\(\):\n"
        r"    x_query = torch\.randn\(batch_size, seq_len_q, dim\)\n"
        r"    x_kv = torch\.randn\(batch_size, seq_len_kv, dim\)\n"
        r"    return \[x_query, x_kv\]\n"
    )
    bound = _init_bound_names(src)
    rep = (
        "def get_inputs():\n"
        f"    p = popcorn_pri\n"
        f"    bs = {_jitter_name('batch_size', bound)}\n"
        f"    tq = {_jitter_name('seq_len_q', bound)}\n"
        f"    tk = {_jitter_name('seq_len_kv', bound)}\n"
        f"    x_query = torch.randn(bs, tq, dim)\n"
        f"    x_kv = torch.randn(bs, tk, dim)\n"
        "    return [x_query, x_kv]\n"
    )
    return pat.sub(rep, src)


def _jitter_evoformer(src: str) -> str:
    pat = re.compile(
        r"def get_inputs\(\):\n"
        r"    msa = torch\.randn\(batch_size, msa_depth, seq_len, msa_dim\)\n"
        r"    pair = torch\.randn\(batch_size, seq_len, seq_len, pair_dim\)\n"
        r"    return \[msa, pair\]\n"
    )
    bound = _init_bound_names(src)
    rep = (
        "def get_inputs():\n"
        f"    p = popcorn_pri\n"
        f"    bs = {_jitter_name('batch_size', bound)}\n"
        f"    sd = {_jitter_name('msa_depth', bound)}\n"
        f"    sl = {_jitter_name('seq_len', bound)}\n"
        f"    msa = torch.randn(bs, sd, sl, msa_dim)\n"
        f"    pair = torch.randn(bs, sl, sl, pair_dim)\n"
        "    return [msa, pair]\n"
    )
    return pat.sub(rep, src)


def _jitter_associative_scan(src: str) -> str:
    pat = re.compile(
        r"def get_inputs\(\):\n"
        r"    a = torch\.sigmoid\(torch\.randn\(batch_size, seq_len, d_model\)\)\n"
        r"    b = torch\.randn\(batch_size, seq_len, d_model\)\n"
        r"    return \[a, b\]\n"
    )
    bound = _init_bound_names(src)
    bs = _jitter_name("batch_size", bound)
    sl = _jitter_name("seq_len", bound)
    rep = (
        f"def get_inputs():\n"
        f"    p = popcorn_pri\n"
        f"    a = torch.sigmoid(torch.randn({bs}, {sl}, d_model))\n"
        f"    b = torch.randn({bs}, {sl}, d_model)\n"
        f"    return [a, b]\n"
    )
    return pat.sub(rep, src)


def _jitter_gaussian_process(src: str) -> str:
    pat = re.compile(
        r"def get_inputs\(\):\n"
        r"    X1 = torch\.randn\(batch_size, num_train, input_dim\)\n"
        r"    X2 = torch\.randn\(batch_size, num_test, input_dim\)\n"
        r"    return \[X1, X2\]\n"
    )
    bound = _init_bound_names(src)
    rep = (
        "def get_inputs():\n"
        f"    p = popcorn_pri\n"
        f"    X1 = torch.randn({_jitter_name('batch_size', bound)}, "
        f"{_jitter_name('num_train', bound)}, input_dim)\n"
        f"    X2 = torch.randn({_jitter_name('batch_size', bound)}, "
        f"{_jitter_name('num_test', bound)}, input_dim)\n"
        "    return [X1, X2]\n"
    )
    return pat.sub(rep, src)


def _jitter_virtual_rank(src: str) -> str:
    if "def get_inputs" not in src or "default_device" not in src:
        return src
    if "popcorn_pri" in src:
        return src
    # Only for problems that define virtual-rank shape constants.
    if not re.search(r"^R\s*=\s*\d+", src, re.M):
        return src
    m = re.search(r"def get_inputs\(\):\n(    .+\n)+", src)
    if not m:
        return src
    block = m.group(0)
    if "torch.randn(R" not in block and "p.jitter_int(R" not in block:
        if "torch.randn(" not in block:
            return src
    uses_dev = "device=dev" in block
    dev_line = "    dev = default_device()\n" if uses_dev else ""
    dev_arg = ", device=dev" if uses_dev else ""
    if "S_MAX" in src and "torch.randn(p.jitter_int(R), p.jitter_int(S_MAX)" not in block:
        new = (
            "def get_inputs():\n"
            f"{dev_line}"
            "    p = popcorn_pri\n"
            f"    return [torch.randn(p.jitter_int(R), p.jitter_int(S_MAX){dev_arg})]\n"
        )
    else:
        new = (
            "def get_inputs():\n"
            f"{dev_line}"
            "    p = popcorn_pri\n"
            f"    return [torch.randn(p.jitter_int(R), p.jitter_int(B), "
            f"p.jitter_int(D, align=8){dev_arg})]\n"
        )
    return src.replace(block, new)


def _jitter_rope_get_inputs(src: str) -> str:
    pat = re.compile(
        r"def get_inputs\(\):\n"
        r"    q = torch\.randn\(batch_size, num_heads, seq_len, dim\)\n"
        r"    k = torch\.randn\(batch_size, num_heads, seq_len, dim\)\n"
        r"    return \[q, k\]\n"
    )
    rep = """def get_inputs():
    p = popcorn_pri
    bs = p.jitter_int(batch_size)
    nh = p.jitter_int(num_heads)
    sl = p.jitter_int(seq_len)
    q = torch.randn(bs, nh, sl, dim)
    k = torch.randn(bs, nh, sl, dim)
    return [q, k]
"""
    return pat.sub(rep, src)


def _jitter_l4_get_inputs(src: str) -> str:
    if "sequence_length" not in src or "batch_size" not in src:
        return src
    if "def get_inputs" not in src:
        return src
    pat = re.compile(r"def get_inputs\(\):\n(?:    .+\n)+", re.MULTILINE)
    if not pat.search(src):
        return src
    return pat.sub(L4_GET_INPUTS + "\n", src, count=1)


def _remove_fixed_generator_seeds(src: str) -> str:
    if "g.manual_seed(" not in src:
        return src
    out_lines: list[str] = []
    skip_until_return = False
    for ln in src.splitlines():
        if re.match(r"\s+g = torch\.Generator", ln):
            skip_until_return = True
            continue
        if skip_until_return:
            if "g.manual_seed" in ln:
                continue
            if "generator=g" in ln:
                ln = ln.replace(", generator=g", "").replace("generator=g, ", "")
            if ln.strip().startswith("return "):
                skip_until_return = False
        out_lines.append(ln)
    return "\n".join(out_lines) + ("\n" if src.endswith("\n") else "")


def _patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "popcorn_pri." in text and "def get_inputs" in text:
        # Allow re-run for files only partially patched
        if "p = popcorn_pri" in text or "popcorn_pri.make_" in text:
            return False
    orig = text
    name = path.name

    if name in {"2_GraphEdgeSoftmaxCSR.py", "8_SegmentTopKCSR.py"}:
        text = _strip_make_helpers(text)
        text = _strip_degree_jitter_lines(text)
        text = _replace_get_inputs_block(text, CSR_SCALAR_EDGE)
    elif name == "6_EdgeSoftmaxMultiHeadCSR.py":
        text = _strip_make_helpers(text)
        text = _strip_degree_jitter_lines(text)
        text = _replace_get_inputs_block(text, CSR_MULTIHEAD_EDGE)
    elif name == "3_CSRSpMMMessagePassing.py":
        text = _strip_make_helpers(text)
        text = _replace_get_inputs_block(text, CSR_SPmm)
    elif name == "11_DegreeNormalizedAggregation.py":
        text = _strip_make_helpers(text)
        text = _replace_get_inputs_block(text, CSR_DEGREE_NORM)
    elif name == "17_CSRMaxAggregation.py":
        text = _strip_degree_jitter_lines(text)
        text = _replace_get_inputs_block(
            text,
            "def get_inputs():\n"
            "    return list(popcorn_pri.make_csr_graph(num_nodes, avg_degree, feat_dim))\n",
        )
    elif name == "18_CSRMultiHeadSpMM.py":
        text = _strip_degree_jitter_lines(text)
        text = _replace_get_inputs_block(text, CSR_MULTIHEAD_SPMM)
    elif name == "37_CSRFusedAttentionValue.py":
        text = _strip_degree_jitter_lines(text)
        text = _replace_get_inputs_block(text, CSR_FUSED_ATTN)
    elif name == "12_DeepSeekMoEDispatchPermute.py":
        text = _replace_get_inputs_block(text, MOE_DISPATCH)
    elif name == "16_DeepSeekMoECombineScatter.py":
        text = _replace_get_inputs_block(text, MOE_COMBINE)
    elif name == "8_DeepSeekMoEGroundedTop2Routing.py":
        text = _replace_get_inputs_block(text, MOE_ROUTING)
    elif name == "15_COOScatterAddNodeFeatures.py":
        text = _replace_get_inputs_block(text, COO_SCATTER)
    elif name == "10_SampledDenseDenseMatmulEdges.py":
        text = _replace_get_inputs_block(text, SAMPLED_MATMUL)

    text = _jitter_l4_get_inputs(text)
    text = _jitter_rope_get_inputs(text)
    text = _jitter_dual_seq_attention(text)
    text = _jitter_evoformer(text)
    text = _jitter_associative_scan(text)
    text = _jitter_gaussian_process(text)
    text = _jitter_virtual_rank(text)
    text = _jitter_conv_get_inputs(text)
    text = _remove_fixed_generator_seeds(text)
    text = _generic_patch_get_inputs(text)

    if text == orig:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    n = 0
    for root in POPCORN_ROOTS:
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
