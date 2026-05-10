"""
Rotary Position Embedding (RoPE) applied to query and key tensors.

NeoxStyle: rotate first half vs second half of head_dim.
Reference: standard complex-number rotation formulation.
"""
import torch
import torch.nn as nn

NUM_TOKENS = 512
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_SIZE = 128
ROT_DIM = 64   # rotary dimension (subset of head_size)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Model(nn.Module):
    def __init__(self, rot_dim: int):
        super().__init__()
        self.rot_dim = rot_dim

    def forward(
        self,
        query: torch.Tensor,   # [num_tokens, num_heads, head_size]
        key: torch.Tensor,     # [num_tokens, num_kv_heads, head_size]
        cos: torch.Tensor,     # [num_tokens, rot_dim]
        sin: torch.Tensor,     # [num_tokens, rot_dim]
    ):
        cos = cos.unsqueeze(1)  # [num_tokens, 1, rot_dim]
        sin = sin.unsqueeze(1)

        q_rot = query[..., : self.rot_dim]
        q_pass = query[..., self.rot_dim :]
        k_rot = key[..., : self.rot_dim]
        k_pass = key[..., self.rot_dim :]

        q_rot = q_rot * cos + _rotate_half(q_rot) * sin
        k_rot = k_rot * cos + _rotate_half(k_rot) * sin

        query_out = torch.cat([q_rot, q_pass], dim=-1)
        key_out = torch.cat([k_rot, k_pass], dim=-1)
        return query_out, key_out


def get_inputs():
    q = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=torch.float16).cuda()
    k = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float16).cuda()
    cos = torch.randn(NUM_TOKENS, ROT_DIM, dtype=torch.float16).cuda()
    sin = torch.randn(NUM_TOKENS, ROT_DIM, dtype=torch.float16).cuda()
    return [q, k, cos, sin]


def get_init_inputs():
    return [ROT_DIM]
