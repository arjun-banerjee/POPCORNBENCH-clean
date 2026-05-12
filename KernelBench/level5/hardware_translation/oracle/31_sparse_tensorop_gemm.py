"""
Structured 2:4 sparse tensor-op GEMM (Ampere / Hopper).

Source problem (A100): kernels/a100/29_ampere_sparse_tensorop_gemm.cu (CUTLASS legacy 15_*)
Source problem (H100): kernels/h100/29_hopper_sparse_gemm.cu (CUTLASS legacy 62_*)

Reference: FP16 matrix multiplication with a resident weight matrix and optional
alpha/beta epilogue. ModelNew may exploit 2:4 structured sparsity in the weight
matrix to use sparse tensor cores on the target GPU, so long as outputs match.
"""
import torch
import torch.nn as nn

M = 5120
N = 4096
K = 16384
ALPHA = 1.0
BETA = 0.0


class Model(nn.Module):
    def __init__(self, k: int, n: int, alpha: float = 1.0, beta: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.register_buffer("weight", torch.randn(k, n, dtype=torch.float16).cuda())
        self.register_buffer("bias", torch.randn(M, n, dtype=torch.float16).cuda())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight
        if self.beta:
            y = self.alpha * y + self.beta * self.bias
        elif self.alpha != 1.0:
            y = self.alpha * y
        return y


def get_inputs():
    return [torch.randn(M, K, dtype=torch.float16).cuda()]


def get_init_inputs():
    return [K, N, ALPHA, BETA]
