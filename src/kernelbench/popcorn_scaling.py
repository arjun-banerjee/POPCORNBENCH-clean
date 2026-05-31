"""Shared hyperparameter tier transforms for popcorn stress / popcorn2."""

from __future__ import annotations

import re

_NEVER_SCALE: frozenset[str] = frozenset(
    {
        "kernel_size",
        "dilation",
        "num_heads",
        "heads",
        "n_heads",
        "topk",
        "k",
        "max_dilation_power",
        "num_layers",
        "num_blocks",
        "rank",
        "stride",
        "padding",
        "groups",
        "d_conv",
        "d_state",
        "expand",
        "pair_dim",
        "head_dim",
        "avg_degree",
        "feat_dim",
        "num_experts",
        "experts_per_token",
        "rope_theta",
        "vocab_size",
        "num_classes",
        "num_bins",
        "sample_size",
        "burn_in",
        "thinning",
        "warmup_steps",
        "num_steps",
        "horizon",
        "order",
        "degree",
        "window_size",
        "patch_size",
        "tile_size",
        "block_size",
        "bucket_size",
        "num_kv_heads",
        "num_attention_heads",
        "n_kv_heads",
    }
)

_SEQ_NAMES: frozenset[str] = frozenset(
    {
        "seq_len",
        "sequence_length",
        "max_seq_len",
        "max_len",
        "length",
        "time_steps",
        "num_tokens",
        "context_length",
        "S_MAX",
    }
)

_MATRIX_NAMES: frozenset[str] = frozenset({"M", "N", "K"})

_BATCH_NAMES: frozenset[str] = frozenset({"batch_size", "batch", "B", "bs"})

_GRAPH_NAMES: frozenset[str] = frozenset({"num_nodes", "num_edges", "num_vertices"})

_CHANNEL_NAMES: frozenset[str] = frozenset(
    {"in_channels", "out_channels", "channels", "hidden_dim", "d_model", "dim"}
)


def has_pair_seq2(src: str) -> bool:
    """O(N^2) pair tensors use seq_len twice (e.g. Evoformer pair stream)."""
    return bool(re.search(r"randn\([^)]*seq_len[^)]*seq_len", src))


def transform_int(
    name: str,
    val: int,
    tier: str,
    *,
    pair_attention: bool = False,
) -> int:
    if name in _NEVER_SCALE:
        return val
    if val < 2:
        return val

    def cap_pair(s: int) -> int:
        lim = {"large": 88, "awkward": 65, "xl": 120}[tier]
        return min(s, lim)

    def cap_seq(s: int) -> int:
        return min(s, 65536)

    def cap_batch(s: int) -> int:
        return min(s, 64)

    def cap_graph_nodes(s: int) -> int:
        return min(s, 6144)

    def cap_matmul(s: int) -> int:
        return min(s, 8192)

    if name in _SEQ_NAMES:
        if pair_attention:
            if tier == "large":
                return cap_pair(max(val + 4, int(val * 1.75)))
            if tier == "awkward":
                return cap_pair(val + (1 if val % 2 == 0 else 2))
            return cap_pair(max(val + 8, int(val * 2.5)))
        if tier == "large":
            return cap_seq(max(val + 16, val * 2))
        if tier == "awkward":
            return cap_seq(val + (1 if val % 2 == 0 else 2))
        return cap_seq(max(val + 32, val * 4))

    if name in _BATCH_NAMES:
        if tier == "large":
            return cap_batch(max(val + 2, val * 2))
        if tier == "awkward":
            return cap_batch(val + (1 if val % 2 == 0 else 2))
        return cap_batch(max(val + 4, val * 2))

    if name in _MATRIX_NAMES:
        if tier == "large":
            return cap_matmul(max(val + 8, val * 2))
        if tier == "awkward":
            return cap_matmul(val + (1 if val % 2 == 0 else 2))
        return cap_matmul(max(val + 16, val * 3))

    if name in _GRAPH_NAMES:
        if tier == "large":
            return cap_graph_nodes(max(val + 16, int(val * 1.35)))
        if tier == "awkward":
            return cap_graph_nodes(val + (1 if val % 2 == 0 else 3))
        return cap_graph_nodes(max(val + 32, int(val * 1.75)))

    if name in _CHANNEL_NAMES and val >= 32:
        if tier == "large":
            return min(512, max(val + 16, int(val * 1.25)))
        if tier == "awkward":
            return min(512, val + (8 if val % 16 == 0 else 7))
        return min(640, max(val + 32, int(val * 1.5)))

    return val
