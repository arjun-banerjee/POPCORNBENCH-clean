"""
SwiGLU activation: silu(gate) * up.

Input tensor has shape [..., 2*d] where the first d elements are the gate
and the second d elements are the up-projection. Output shape is [..., d].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_TOKENS = 512
D = 2048   # output dim; input is 2*D


class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [num_tokens, 2*d]
        d = x.shape[-1] // 2
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up


def get_inputs():
    x = torch.randn(NUM_TOKENS, 2 * D, dtype=torch.float16).cuda()
    return [x]


def get_init_inputs():
    return []
