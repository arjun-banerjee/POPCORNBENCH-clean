"""
Turing INT8 tensor-op GEMM.

Reference: cast activations and weights to int32 and run matrix multiplication.
"""
import torch
import torch.nn as nn

M = 5120
N = 4096
K = 4096
ALPHA = 1
BETA = 0


class Model(nn.Module):
    def __init__(self, k: int, n: int, alpha: int = 1, beta: int = 0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.register_buffer("weight", torch.randint(-4, 5, (k, n), dtype=torch.int8).cuda())
        self.register_buffer("bias", torch.randint(-4, 5, (M, n), dtype=torch.int32).cuda())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.to(torch.int32) @ self.weight.to(torch.int32)
        if self.beta:
            y = self.alpha * y + self.beta * self.bias
        elif self.alpha != 1:
            y = self.alpha * y
        return y


def get_inputs():
    return [torch.randint(-4, 5, (M, K), dtype=torch.int8).cuda()]


def get_init_inputs():
    return [K, N, ALPHA, BETA]

