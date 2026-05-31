# popcorn2: large-tier module centers (scripts/gen_popcorn2_centers.py).
# Source: KernelBench/level1/popcorn/30_broadcast_masked_from_row0.py

"""
Simulated broadcast of row 0 into masked columns for all rows: where mask[j] is True,
every row matches x[0, j]; elsewhere keep x[b, j].
"""
import torch
import torch.nn as nn
from kernelbench.distributed_collectives import default_device
B = 36
D = 44
_MASK = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1], dtype=torch.bool)

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('row_mask', _MASK.clone(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.row_mask.to(device=x.device).view(1, D).expand(B, D)
        row0 = x[0:1].expand(B, D)
        return torch.where(m, row0, x)

def get_inputs():
    p = popcorn_pri
    mode = p.sample_input_mode()
    dev = default_device()
    return [torch.randn(p.trial_dim(B, 'B', mode=mode), D, device=dev)]

def get_init_inputs():
    return []
