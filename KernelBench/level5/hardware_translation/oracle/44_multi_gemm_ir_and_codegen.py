"""
A100 generated back-to-back GEMM pipeline.

Reference: three FP16 GEMMs with LeakyReLU epilogues, using the shapes from
44_multi_gemm_ir_and_codegen/config.json.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

M = 15000
K0 = 32
N0 = 256
N1 = 128
N2 = 64
LEAKY_ALPHA = 1.3


class Model(nn.Module):
    def __init__(
        self,
        k0: int,
        n0: int,
        n1: int,
        n2: int,
        leaky_alpha: float = 1.3,
    ):
        super().__init__()
        self.leaky_alpha = leaky_alpha
        self.register_buffer("weight0", torch.randn(k0, n0, dtype=torch.float16).cuda())
        self.register_buffer("weight1", torch.randn(n0, n1, dtype=torch.float16).cuda())
        self.register_buffer("weight2", torch.randn(n1, n2, dtype=torch.float16).cuda())

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=self.leaky_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._act(x @ self.weight0)
        x = self._act(x @ self.weight1)
        return self._act(x @ self.weight2)


def get_inputs():
    return [torch.randn(M, K0, dtype=torch.float16).cuda()]


def get_init_inputs():
    return [K0, N0, N1, N2, LEAKY_ALPHA]

