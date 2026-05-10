"""
A100 batched GEMM.

Reference: one independent FP32 GEMM per batch item with alpha/beta epilogue.
"""
import torch
import torch.nn as nn

BATCH = 17
M = 520
N = 219
K = 129
ALPHA = 1.0
BETA = 2.0


class Model(nn.Module):
    def __init__(
        self,
        batch: int,
        k: int,
        n: int,
        alpha: float = 1.0,
        beta: float = 2.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.register_buffer("weight", torch.randn(batch, k, n, dtype=torch.float32).cuda())
        self.register_buffer("bias", torch.ones(batch, M, n, dtype=torch.float32).cuda())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.bmm(x, self.weight) + self.beta * self.bias


def get_inputs():
    return [torch.randn(BATCH, M, K, dtype=torch.float32).cuda()]


def get_init_inputs():
    return [BATCH, K, N, ALPHA, BETA]

