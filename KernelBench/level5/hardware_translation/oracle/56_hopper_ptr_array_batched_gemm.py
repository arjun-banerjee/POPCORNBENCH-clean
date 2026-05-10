"""
Hopper pointer-array batched GEMM.

Source A100 kernel: 05_batched_gemm.
Target H100 kernel: 56_hopper_ptr_array_batched_gemm.

Reference: one independent FP32 GEMM per batch item.
"""
import torch
import torch.nn as nn

BATCH = 10
M = 1024
N = 512
K = 1024
ALPHA = 1.0
BETA = 0.0


class Model(nn.Module):
    def __init__(
        self,
        batch: int,
        k: int,
        n: int,
        alpha: float = 1.0,
        beta: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.register_buffer("weight", torch.randn(batch, k, n, dtype=torch.float32).cuda())
        self.register_buffer("bias", torch.randn(batch, M, n, dtype=torch.float32).cuda())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.bmm(x, self.weight)
        if self.beta:
            y = self.alpha * y + self.beta * self.bias
        elif self.alpha != 1.0:
            y = self.alpha * y
        return y


def get_inputs():
    return [torch.randn(BATCH, M, K, dtype=torch.float32).cuda()]


def get_init_inputs():
    return [BATCH, K, N, ALPHA, BETA]

