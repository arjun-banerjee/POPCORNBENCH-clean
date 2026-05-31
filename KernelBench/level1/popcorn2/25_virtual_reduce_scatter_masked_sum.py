# popcorn2: large-tier module centers (scripts/gen_popcorn2_centers.py).
# Source: KernelBench/level1/popcorn/25_virtual_reduce_scatter_masked_sum.py

"""
Unequal virtual shard lengths: x is (R, S_MAX) with a fixed boolean mask marking valid cells.
Output (S_MAX,) is sum_r x[r, j] * mask[r, j] (masked sum along the virtual rank dimension).

Padding positions are masked out so they do not contribute.
"""
import torch
import torch.nn as nn
from kernelbench.distributed_collectives import default_device
R = 4
S_MAX = 40
_MASK_DATA = [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('shard_mask', torch.tensor(_MASK_DATA, dtype=torch.bool), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.shard_mask.to(dtype=x.dtype, device=x.device)
        return (x * m).sum(dim=0)

def get_inputs():
    p = popcorn_pri
    mode = p.sample_input_mode()
    dev = default_device()
    return [torch.randn(p.trial_dim(R, 'R', mode=mode), p.trial_dim(S_MAX, 'S_MAX', mode=mode), device=dev)]

def get_init_inputs():
    return []
