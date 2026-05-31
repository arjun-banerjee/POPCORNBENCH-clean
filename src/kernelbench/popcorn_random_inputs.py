"""
Per-trial randomization for KernelBench popcorn reference ``get_inputs()``.

Eval sets ``torch.manual_seed(trial_seed)`` before each ``get_inputs()`` call.
Helpers here use the global PyTorch RNG so CSR layouts, graph sizes, and tensor
shapes change across correctness trials — kernels must read ``row_ptr`` / shapes
from tensors, not hardcode closed-form structure.

``popcorn2`` problems use large-tier module constants plus ``sample_input_mode`` /
``trial_dim`` for biased large-band / awkward / wide per-trial shapes.
"""

from __future__ import annotations

import os

import torch

from kernelbench.popcorn_scaling import transform_int

# Injected into problem ``exec`` context as ``popcorn_pri`` (see eval.py).

_DEFAULT_P_LARGE_BAND = 0.60
_DEFAULT_P_AWKWARD = 0.30
_DEFAULT_P_WIDE = 0.10
_DEFAULT_JITTER_MIN = 0.90
_DEFAULT_JITTER_MAX = 1.10
_DEFAULT_WIDE_MIN = 0.80
_DEFAULT_WIDE_MAX = 1.20


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def trial_int(low: int, high: int) -> int:
    if low > high:
        low, high = high, low
    return int(torch.randint(low, high + 1, (1,), dtype=torch.int64).item())


def jitter_int(
    default: int,
    *,
    min_ratio: float = 0.85,
    max_ratio: float = 1.15,
    minimum: int = 1,
    align: int = 1,
) -> int:
    lo = max(minimum, int(default * min_ratio))
    hi = max(lo, int(default * max_ratio))
    v = trial_int(lo, hi)
    if align > 1:
        v = max(align, (v // align) * align)
    return v


def sample_input_mode(
    *,
    p_large_band: float | None = None,
    p_awkward: float | None = None,
    p_wide: float | None = None,
) -> str:
    """Pick per-trial shape mode: ``large_band``, ``awkward``, or ``wide``."""
    pl = p_large_band if p_large_band is not None else _env_float(
        "POPCORN2_P_LARGE_BAND", _DEFAULT_P_LARGE_BAND
    )
    pa = p_awkward if p_awkward is not None else _env_float(
        "POPCORN2_P_AWKWARD", _DEFAULT_P_AWKWARD
    )
    pw = p_wide if p_wide is not None else _env_float("POPCORN2_P_WIDE", _DEFAULT_P_WIDE)
    total = pl + pa + pw
    if total <= 0:
        return "large_band"
    pl, pa, pw = pl / total, pa / total, pw / total
    r = float(torch.rand(()).item())
    if r < pl:
        return "large_band"
    if r < pl + pa:
        return "awkward"
    return "wide"


def trial_dim(
    center: int,
    name: str = "",
    *,
    mode: str | None = None,
    min_ratio: float | None = None,
    max_ratio: float | None = None,
    minimum: int = 1,
    align: int = 1,
    pair_attention: bool = False,
) -> int:
    """
    Biased per-trial dimension for ``popcorn2``: jitter around a large-tier center,
    or snap to awkward-tier transform rules.
    """
    if mode is None:
        mode = sample_input_mode()
    dim_name = name or "_"
    if mode == "awkward":
        v = transform_int(
            dim_name, center, "awkward", pair_attention=pair_attention
        )
    elif mode == "wide":
        lo = _env_float("POPCORN2_JITTER_WIDE_MIN", _DEFAULT_WIDE_MIN)
        hi = _env_float("POPCORN2_JITTER_WIDE_MAX", _DEFAULT_WIDE_MAX)
        v = jitter_int(
            center,
            min_ratio=lo,
            max_ratio=hi,
            minimum=minimum,
            align=align,
        )
    else:
        lo = min_ratio if min_ratio is not None else _env_float(
            "POPCORN2_JITTER_MIN", _DEFAULT_JITTER_MIN
        )
        hi = max_ratio if max_ratio is not None else _env_float(
            "POPCORN2_JITTER_MAX", _DEFAULT_JITTER_MAX
        )
        v = jitter_int(
            center,
            min_ratio=lo,
            max_ratio=hi,
            minimum=minimum,
            align=align,
        )
    return v


def _trial_n(
    center: int,
    name: str,
    *,
    mode: str | None,
    align: int = 1,
) -> int:
    if mode is None:
        return jitter_int(center, align=align) if align > 1 else jitter_int(center)
    return trial_dim(center, name, mode=mode, align=align)


def random_node_degrees(
    num_nodes: int,
    avg_degree: int,
    *,
    min_degree: int = 1,
    spread: int = 8,
) -> torch.Tensor:
    lo = max(min_degree, avg_degree - spread)
    hi = avg_degree + spread
    return torch.randint(lo, hi + 1, (num_nodes,), dtype=torch.int32)


def csr_row_ptr_from_degrees(degree: torch.Tensor) -> torch.Tensor:
    row_ptr = torch.zeros(degree.numel() + 1, dtype=torch.int32)
    row_ptr[1:] = torch.cumsum(degree, dim=0)
    return row_ptr


def random_csr_row_ptr(num_nodes: int, avg_degree: int) -> torch.Tensor:
    return csr_row_ptr_from_degrees(random_node_degrees(num_nodes, avg_degree))


def sample_multihead_dims(
    num_heads_default: int,
    head_dim_default: int,
    *,
    head_choices: tuple[int, ...] = (4, 8, 16),
    dim_choices: tuple[int, ...] = (16, 32, 64),
) -> tuple[int, int]:
    """Pick head count / head dim from nearby standard values (varies per trial)."""
    nh_pool = [h for h in head_choices if abs(h - num_heads_default) <= 8]
    hd_pool = [d for d in dim_choices if abs(d - head_dim_default) <= 32]
    if not nh_pool:
        nh_pool = list(head_choices)
    if not hd_pool:
        hd_pool = list(dim_choices)
    nh = nh_pool[trial_int(0, len(nh_pool) - 1)]
    hd = hd_pool[trial_int(0, len(hd_pool) - 1)]
    return nh, hd


def make_csr_graph(
    num_nodes_default: int,
    avg_degree_default: int,
    feat_dim_default: int,
    *,
    jitter_nodes: bool = True,
    jitter_feat: bool = True,
    mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if jitter_nodes:
        n = _trial_n(num_nodes_default, "num_nodes", mode=mode)
    else:
        n = num_nodes_default
    row_ptr = random_csr_row_ptr(n, avg_degree_default)
    num_edges = int(row_ptr[-1].item())
    col_idx = torch.randint(0, n, (num_edges,), dtype=torch.int32)
    if jitter_feat:
        fd = _trial_n(feat_dim_default, "feat_dim", mode=mode, align=8)
    else:
        fd = feat_dim_default
    node_feat = torch.randn(n, fd, dtype=torch.float32)
    return row_ptr, col_idx, node_feat


def make_csr_graph_with_edge_weights(
    num_nodes_default: int,
    avg_degree_default: int,
    feat_dim_default: int,
    *,
    mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row_ptr, col_idx, node_feat = make_csr_graph(
        num_nodes_default, avg_degree_default, feat_dim_default, mode=mode
    )
    num_edges = int(row_ptr[-1].item())
    edge_weight = torch.randn(num_edges, dtype=torch.float32)
    return row_ptr, col_idx, edge_weight, node_feat


def make_csr_degree_normalized_graph(
    num_nodes_default: int,
    avg_degree_default: int,
    feat_dim_default: int,
    *,
    mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row_ptr, col_idx, node_feat = make_csr_graph(
        num_nodes_default, avg_degree_default, feat_dim_default, mode=mode
    )
    n = int(row_ptr.numel()) - 1
    degrees = torch.bincount(col_idx.long(), minlength=n).to(torch.float32) + 1.0
    return row_ptr, col_idx, node_feat, degrees


def make_csr_multihead_spmm_graph(
    num_nodes_default: int,
    avg_degree_default: int,
    num_heads_default: int,
    head_dim_default: int,
    *,
    mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = _trial_n(num_nodes_default, "num_nodes", mode=mode)
    nh, hd = sample_multihead_dims(num_heads_default, head_dim_default)
    row_ptr = random_csr_row_ptr(n, avg_degree_default)
    num_edges = int(row_ptr[-1].item())
    col_idx = torch.randint(0, n, (num_edges,), dtype=torch.int32)
    edge_weight = torch.randn(num_edges, nh, dtype=torch.float32)
    node_feat = torch.randn(n, nh, hd, dtype=torch.float32)
    return row_ptr, col_idx, edge_weight, node_feat


def make_csr_scalar_edge_scores(
    num_nodes_default: int,
    avg_degree_default: int,
    *,
    mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = _trial_n(num_nodes_default, "num_nodes", mode=mode)
    row_ptr = random_csr_row_ptr(n, avg_degree_default)
    num_edges = int(row_ptr[-1].item())
    edge_scores = torch.randn(num_edges, dtype=torch.float32)
    return row_ptr, edge_scores


def make_csr_multihead_edge_scores(
    num_nodes_default: int,
    avg_degree_default: int,
    num_heads_default: int,
    *,
    mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = _trial_n(num_nodes_default, "num_nodes", mode=mode)
    nh, _ = sample_multihead_dims(num_heads_default, 32)
    row_ptr = random_csr_row_ptr(n, avg_degree_default)
    num_edges = int(row_ptr[-1].item())
    edge_scores = torch.randn(num_edges, nh, dtype=torch.float32)
    return row_ptr, edge_scores


def make_csr_fused_attention_value_graph(
    num_nodes_default: int,
    avg_degree_default: int,
    feat_dim_default: int,
    *,
    mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row_ptr, col_idx, _ = make_csr_graph(
        num_nodes_default, avg_degree_default, feat_dim_default, mode=mode
    )
    num_edges = int(row_ptr[-1].item())
    edge_scores = torch.randn(num_edges, dtype=torch.float32)
    n = int(row_ptr.numel()) - 1
    fd = _trial_n(feat_dim_default, "feat_dim", mode=mode, align=8)
    node_value = torch.randn(n, fd, dtype=torch.float32)
    return row_ptr, col_idx, edge_scores, node_value


def moe_dispatch_inputs(
    num_tokens_default: int,
    hidden_dim_default: int,
    num_experts_default: int,
    *,
    mode: str | None = None,
) -> list[torch.Tensor]:
    num_tokens = _trial_n(num_tokens_default, "num_tokens", mode=mode, align=64)
    hidden_dim = _trial_n(hidden_dim_default, "hidden_dim", mode=mode, align=16)
    num_experts = trial_dim(
        num_experts_default, "num_experts", mode=mode, minimum=2, align=1
    )
    num_experts = max(2, min(num_experts, num_tokens))

    token_hidden = torch.randn(num_tokens, hidden_dim, dtype=torch.float32)
    expert_idx = torch.randint(0, num_experts, (num_tokens,), dtype=torch.int32)
    counts = torch.bincount(expert_idx.to(torch.int64), minlength=num_experts).to(
        torch.int32
    )
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32)
    expert_offsets[1:] = torch.cumsum(counts, dim=0)
    slot_cursor = torch.zeros(num_experts, dtype=torch.int32)
    slot_idx = torch.empty(num_tokens, dtype=torch.int32)
    for token in range(num_tokens):
        expert = int(expert_idx[token].item())
        slot_idx[token] = slot_cursor[expert]
        slot_cursor[expert] += 1
    return [token_hidden, expert_idx, slot_idx, expert_offsets]


def moe_combine_inputs(
    num_tokens_default: int,
    hidden_dim_default: int,
    fanout_default: int,
    *,
    mode: str | None = None,
) -> list[torch.Tensor]:
    num_tokens = _trial_n(num_tokens_default, "num_tokens", mode=mode, align=64)
    hidden_dim = _trial_n(hidden_dim_default, "hidden_dim", mode=mode, align=16)
    fanout = trial_dim(fanout_default, "fanout", mode=mode, minimum=1, align=1)

    expert_hidden = torch.randn(num_tokens * fanout, hidden_dim, dtype=torch.float32)
    token_idx = torch.randint(0, num_tokens, (num_tokens * fanout,), dtype=torch.int32)
    gates = torch.softmax(
        torch.randn(num_tokens, fanout, dtype=torch.float32), dim=-1
    ).reshape(-1)
    return [expert_hidden, token_idx, gates, num_tokens]
