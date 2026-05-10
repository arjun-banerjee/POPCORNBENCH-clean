"""
Paged Attention V2: two-pass paged attention with partition-level parallelism.

Reference: same GQA attention as V1; the two-pass partition trick is an
implementation detail tested via performance, not a functional change.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_SIZE = 128
SEQ_LEN = 1024
BATCH = 4


class Model(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = head_size ** -0.5
        self.groups = num_heads // num_kv_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        key = key.repeat_interleave(self.groups, dim=1)
        value = value.repeat_interleave(self.groups, dim=1)
        return F.scaled_dot_product_attention(
            query, key, value, scale=self.scale, is_causal=True
        )


def get_inputs():
    q = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_SIZE,
                    dtype=torch.float16).cuda()
    k = torch.randn(BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_SIZE,
                    dtype=torch.float16).cuda()
    v = torch.randn(BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_SIZE,
                    dtype=torch.float16).cuda()
    return [q, k, v]


def get_init_inputs():
    return [NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE]
