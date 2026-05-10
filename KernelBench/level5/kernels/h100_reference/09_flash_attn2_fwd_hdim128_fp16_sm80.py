"""
Flash Attention Forward Pass (causal).

A100 kernel: FlashAttention-2 (SM80, FP16, head_dim=128).
H100 kernel: FlashAttention-3 (SM90, BF16, head_dim=128).

Reference: torch.nn.functional.scaled_dot_product_attention (causal mask).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH = 4
NUM_HEADS = 32
NUM_KV_HEADS = 8
SEQ_LEN = 2048
HEAD_DIM = 128


class Model(nn.Module):
    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,  # [batch, num_heads, seqlen, head_dim]
        k: torch.Tensor,  # [batch, num_kv_heads, seqlen, head_dim]
        v: torch.Tensor,  # [batch, num_kv_heads, seqlen, head_dim]
    ) -> torch.Tensor:
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        return F.scaled_dot_product_attention(
            q, k, v, scale=self.scale, is_causal=True
        )


def get_inputs():
    dtype = torch.bfloat16
    q = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype).cuda()
    k = torch.randn(BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype).cuda()
    v = torch.randn(BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype).cuda()
    return [q, k, v]


def get_init_inputs():
    return [NUM_HEADS, NUM_KV_HEADS, HEAD_DIM]
