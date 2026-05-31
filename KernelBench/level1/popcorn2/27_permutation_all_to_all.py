# popcorn2: large-tier module centers (scripts/gen_popcorn2_centers.py).
# Source: KernelBench/level1/popcorn/27_permutation_all_to_all.py

"""
Fixed permutation along the feature axis: y[b, j] = x[b, perm[j]].
"""
import torch
import torch.nn as nn
from kernelbench.distributed_collectives import default_device
B = 20
N = 72
_g = torch.Generator()
_g.manual_seed(10)
PERM = torch.randperm(N, generator=_g)

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, PERM.to(device=x.device)]

def get_inputs():
    p = popcorn_pri
    mode = p.sample_input_mode()
    dev = default_device()
    return [torch.randn(p.trial_dim(B, 'B', mode=mode), N, device=dev)]

def get_init_inputs():
    return []
