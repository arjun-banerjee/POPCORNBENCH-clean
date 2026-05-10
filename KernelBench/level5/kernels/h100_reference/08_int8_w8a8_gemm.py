"""
Quantized W8A8 GEMM.

A100 kernel: INT8 weights and activations (w8a8 INT8).
H100 kernel: FP8 weights and activations (w8a8 FP8/e4m3).

Reference: cast activations and weights to float16, run torch.matmul.
This validates functional correctness; precision differences vs a quantized
reference are expected and should be checked with a relaxed tolerance.
"""
import torch
import torch.nn as nn

M = 16
N = 4096
K = 4096


class Model(nn.Module):
    def __init__(self, k: int, n: int):
        super().__init__()
        # Weight in INT8; scale per output channel
        self.register_buffer(
            "weight", torch.randint(-128, 127, (n, k), dtype=torch.int8).cuda()
        )
        self.register_buffer(
            "weight_scale", torch.ones(n, dtype=torch.float32).cuda() / 127.0
        )
        self.register_buffer(
            "act_scale", torch.ones(1, dtype=torch.float32).cuda() / 127.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, K] float16/bfloat16
        # Dequantize weight
        w_fp = self.weight.to(torch.float32) * self.weight_scale.unsqueeze(-1)
        x_fp = x.to(torch.float32)
        return (x_fp @ w_fp.t()).to(x.dtype)


def get_inputs():
    return [torch.randn(M, K, dtype=torch.float16).cuda()]


def get_init_inputs():
    return [K, N]
