"""
Blackwell fused multi-head attention.

Source A100 kernel: 44_multi_gemm_ir_and_codegen.
Target H100/Blackwell kernel: 77_blackwell_fmha.

Reference: PyTorch scaled dot-product attention with a causal mask.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH = 2
NUM_HEADS = 16
SEQ_LEN = 1024
HEAD_DIM = 128


class Model(nn.Module):
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, scale=self.scale, is_causal=True)


def get_inputs():
    dtype = torch.bfloat16
    q = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype).cuda()
    k = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype).cuda()
    v = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype).cuda()
    return [q, k, v]


def get_init_inputs():
    return [NUM_HEADS, HEAD_DIM]

