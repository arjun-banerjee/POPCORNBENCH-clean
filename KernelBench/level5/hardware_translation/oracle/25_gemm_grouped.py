"""
A100 grouped GEMM.

Reference: a uniform grouped workload represented as batched FP32 GEMM.
"""
import torch
import torch.nn as nn

GROUPS = 15
M = 256
N = 256
K = 256
ALPHA = 1.0
BETA = 0.0


class Model(nn.Module):
    def __init__(
        self,
        groups: int,
        k: int,
        n: int,
        alpha: float = 1.0,
        beta: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.register_buffer("weight", torch.randn(groups, k, n, dtype=torch.float32).cuda())
        self.register_buffer("bias", torch.randn(groups, M, n, dtype=torch.float32).cuda())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.bmm(x, self.weight)
        if self.beta:
            y = self.alpha * y + self.beta * self.bias
        elif self.alpha != 1.0:
            y = self.alpha * y
        return y


def get_inputs():
    return [torch.randn(GROUPS, M, K, dtype=torch.float32).cuda()]


def get_init_inputs():
    return [GROUPS, K, N, ALPHA, BETA]

