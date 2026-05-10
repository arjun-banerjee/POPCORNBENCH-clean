"""
Hopper warp-specialized TF32 GEMM.

Source A100 kernel: 14_ampere_tf32_tensorop_gemm.
Target H100 kernel: 48_hopper_warp_specialized_gemm.

Reference: FP32 matrix multiplication with a resident weight matrix.
"""
import torch
import torch.nn as nn

M = 5120
N = 4096
K = 4096
ALPHA = 1.0
BETA = 0.0


class Model(nn.Module):
    def __init__(self, k: int, n: int, alpha: float = 1.0, beta: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.register_buffer("weight", torch.randn(k, n, dtype=torch.float32).cuda())
        self.register_buffer("bias", torch.randn(M, n, dtype=torch.float32).cuda())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight
        if self.beta:
            y = self.alpha * y + self.beta * self.bias
        elif self.alpha != 1.0:
            y = self.alpha * y
        return y


def get_inputs():
    return [torch.randn(M, K, dtype=torch.float32).cuda()]


def get_init_inputs():
    return [K, N, ALPHA, BETA]

