# popcorn2: large-tier module centers (scripts/gen_popcorn2_centers.py).
# Source: KernelBench/level1/popcorn/23_virtual_all_reduce_mean.py

"""
Virtual ranks: input packs R per-rank tensors along dim 0. Output is the mean over ranks (B, D).

No distributed runtime required; the reference always reduces along dim 0.
"""
import torch
import torch.nn as nn
from kernelbench.distributed_collectives import default_device
R = 4
B = 32
D = 48

class Model(nn.Module):
    """x shape (R, B, D) -> mean over R -> (B, D)."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=0)

def get_inputs():
    p = popcorn_pri
    mode = p.sample_input_mode()
    dev = default_device()
    return [torch.randn(p.trial_dim(R, 'R', mode=mode), p.trial_dim(B, 'B', mode=mode), p.trial_dim(D, 'D', mode=mode, align=8), device=dev)]

def get_init_inputs():
    return []
