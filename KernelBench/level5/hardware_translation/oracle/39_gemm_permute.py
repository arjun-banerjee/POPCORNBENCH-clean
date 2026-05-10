"""
A100 GEMM with output permutation.

Reference: batched FP16 GEMM followed by Tensor4DPermuteBMM0213-style output layout.
"""
import torch
import torch.nn as nn

BATCH = 96
M = 384
N = 192
K = 384
D1 = 12
ALPHA = 1.0
BETA = 0.0


class Model(nn.Module):
    def __init__(
        self,
        batch: int,
        k: int,
        n: int,
        d1: int,
        alpha: float = 1.0,
        beta: float = 0.0,
    ):
        super().__init__()
        self.batch = batch
        self.n = n
        self.d1 = d1
        self.alpha = alpha
        self.beta = beta
        self.register_buffer("weight", torch.randn(batch, k, n, dtype=torch.float16).cuda())
        self.register_buffer("bias", torch.randn(batch, M, n, dtype=torch.float16).cuda())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.bmm(x, self.weight)
        if self.beta:
            y = self.alpha * y + self.beta * self.bias
        elif self.alpha != 1.0:
            y = self.alpha * y
        return y.view(self.batch // self.d1, self.d1, M, self.n).permute(
            0, 2, 1, 3
        ).contiguous()


def get_inputs():
    return [torch.randn(BATCH, M, K, dtype=torch.float16).cuda()]


def get_init_inputs():
    return [BATCH, K, N, D1, ALPHA, BETA]

