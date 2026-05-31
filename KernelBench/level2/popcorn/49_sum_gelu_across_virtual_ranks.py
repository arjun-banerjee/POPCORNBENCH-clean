"""
Order-sensitive virtual collective: out = sum_r GELU(x_r), same shape as one rank slice (B, D).

Note: this is not equal to GELU(sum_r x_r).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernelbench.distributed_collectives import default_device

R = 4
B = 14
D = 40


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x).sum(dim=0)


def get_inputs():
    dev = default_device()
    p = popcorn_pri
    return [torch.randn(p.jitter_int(R), p.jitter_int(B), p.jitter_int(D, align=8), device=dev)]


def get_init_inputs():
    return []
