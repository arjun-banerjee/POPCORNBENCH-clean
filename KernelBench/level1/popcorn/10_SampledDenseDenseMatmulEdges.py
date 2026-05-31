import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Edge-only sampled dense-dense matmul.

    For each graph edge (src, dst), compute a dot product between source and
    destination node embeddings. This pattern appears in link prediction,
    sampled attention, and graph contrastive learning.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        src_feat: torch.Tensor,
        dst_feat: torch.Tensor,
    ) -> torch.Tensor:
        src = src_feat[src_idx.long()]
        dst = dst_feat[dst_idx.long()]
        return (src * dst).sum(dim=-1)


num_src_nodes = 1024
num_dst_nodes = 768
num_edges = 16384
feat_dim = 128


def get_inputs():
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





def get_init_inputs():
    return []
