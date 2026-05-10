"""
Hopper GEMM with output permutation.

Source A100 kernel: 39_gemm_permute.
Target H100 kernel: 53_hopper_gemm_permute.

Reference: batched FP16 GEMM followed by Tensor4DPermute0213-style output layout.
"""
import torch
import torch.nn as nn

BATCH = 8
M = 2048
N = 2048
K = 2048
D1 = 8
D2 = 16
ALPHA = 1.0
BETA = 0.0


class Model(nn.Module):
    def __init__(
        self,
        batch: int,
        k: int,
        n: int,
        d1: int,
        d2: int,
        alpha: float = 1.0,
        beta: float = 0.0,
    ):
        super().__init__()
        self.batch = batch
        self.n = n
        self.d1 = d1
        self.d2 = d2
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
        return y.view(self.batch, M // self.d1, self.d1, self.d2, self.n // self.d2).permute(
            0, 1, 3, 2, 4
        ).contiguous()


def get_inputs():
    return [torch.randn(BATCH, M, K, dtype=torch.float16).cuda()]


def get_init_inputs():
    return [BATCH, K, N, D1, D2, ALPHA, BETA]

