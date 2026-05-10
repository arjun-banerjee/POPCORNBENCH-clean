"""
Flash Attention Backward Pass (causal).

A100 kernel: FlashAttention-2 backward (SM80, FP16, head_dim=128).
H100 kernel: FlashAttention-3 backward (SM90, BF16, head_dim=128).

Reference: autograd through torch SDPA.
NOTE: The backward pass returns dQ, dK, dV gradients given dO (upstream grad).
      Model.forward returns the attention output so autograd can be triggered.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH = 2
NUM_HEADS = 32
NUM_KV_HEADS = 8
SEQ_LEN = 1024
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
        q: torch.Tensor,    # [batch, num_heads, seqlen, head_dim]
        k: torch.Tensor,
        v: torch.Tensor,
        dout: torch.Tensor, # [batch, num_heads, seqlen, head_dim] upstream grad
    ) -> tuple:
        # Re-enable grad so autograd.grad works even inside no_grad contexts
        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            k = k.detach().requires_grad_(True)
            v = v.detach().requires_grad_(True)
            k_exp = k.repeat_interleave(self.groups, dim=1)
            v_exp = v.repeat_interleave(self.groups, dim=1)
            out = F.scaled_dot_product_attention(
                q, k_exp, v_exp, scale=self.scale, is_causal=True
            )
            grads = torch.autograd.grad(out, [q, k, v], grad_outputs=dout)
        return grads  # (dq, dk, dv)


def get_inputs():
    dtype = torch.bfloat16
    q = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM,
                    dtype=dtype, requires_grad=True).cuda()
    k = torch.randn(BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM,
                    dtype=dtype, requires_grad=True).cuda()
    v = torch.randn(BATCH, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM,
                    dtype=dtype, requires_grad=True).cuda()
    dout = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype).cuda()
    return [q, k, v, dout]


def get_init_inputs():
    return [NUM_HEADS, NUM_KV_HEADS, HEAD_DIM]
