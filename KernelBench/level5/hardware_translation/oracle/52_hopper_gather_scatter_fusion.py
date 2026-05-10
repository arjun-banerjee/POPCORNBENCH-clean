"""
Hopper gather-scatter GEMM fusion.

Source A100 kernel: 36_gather_scatter_fusion.
Target H100 kernel: 52_hopper_gather_scatter_fusion.

Reference: gather selected N columns, run GEMM, then scatter into a full output.
"""
import torch
import torch.nn as nn

BATCH = 1
M = 2048
N = 2048
K = 2048
INDEX_SIZE = 1024
ALPHA = 1.0
BETA = 0.0


class Model(nn.Module):
    def __init__(
        self,
        k: int,
        n: int,
        index_size: int,
        alpha: float = 1.0,
        beta: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        indices = torch.randperm(n, device="cuda", dtype=torch.int64)[:index_size]
        self.register_buffer("indices", indices)
        self.register_buffer("weight", torch.randn(k, n, dtype=torch.float16).cuda())
        self.register_buffer("bias", torch.randn(BATCH, M, n, dtype=torch.float32).cuda())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gathered_weight = self.weight.index_select(1, self.indices)
        partial = (x @ gathered_weight).to(torch.float32)
        out = torch.zeros(BATCH, M, N, device=x.device, dtype=torch.float32)
        out.index_copy_(2, self.indices, self.alpha * partial)
        if self.beta:
            out = out + self.beta * self.bias
        return out


def get_inputs():
    return [torch.randn(BATCH, M, K, dtype=torch.float16).cuda()]


def get_init_inputs():
    return [K, N, INDEX_SIZE, ALPHA, BETA]

