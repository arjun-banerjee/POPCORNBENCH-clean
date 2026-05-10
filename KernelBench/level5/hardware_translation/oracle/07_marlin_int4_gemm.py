"""
INT4 Weight-only Quantized GEMM.

A100 kernel: Marlin format (sparse INT4 with special tile layout).
H100 kernel: Machete format (SM90-native INT4 with wgmma).

Reference: dequantize INT4 weights to FP16 then run torch.matmul.
Pack format: weights are grouped-quantized INT4, 2 values per byte,
using a scale-and-zero-point scheme (group_size=128).
"""
import torch
import torch.nn as nn

M = 16        # batch / token count
N = 4096      # output features
K = 4096      # input features
GROUP_SIZE = 128


def dequantize_int4(
    qweight: torch.Tensor,   # [K, N//2]  packed uint8
    scales: torch.Tensor,    # [K//group_size, N]
    zeros: torch.Tensor,     # [K//group_size, N]  zero-points
) -> torch.Tensor:
    K_, Nhalf = qweight.shape
    N_ = Nhalf * 2
    # Unpack two INT4s from each byte
    lo = (qweight & 0x0F).to(torch.float16)  # lower nibble
    hi = (qweight >> 4).to(torch.float16)    # upper nibble
    # Interleave: [K, N//2] -> [K, N]
    weight_f = torch.stack([lo, hi], dim=-1).reshape(K_, N_)
    # Apply grouped scale/zero
    num_groups = K_ // GROUP_SIZE
    weight_f = weight_f.view(num_groups, GROUP_SIZE, N_)
    scales_ = scales.unsqueeze(1)   # [num_groups, 1, N]
    zeros_ = zeros.unsqueeze(1)
    weight_f = (weight_f - zeros_) * scales_
    return weight_f.view(K_, N_)


class Model(nn.Module):
    def __init__(self, k: int, n: int, group_size: int = 128):
        super().__init__()
        num_groups = k // group_size
        # Packed uint8: [K, N//2]
        self.register_buffer(
            "qweight",
            torch.randint(0, 256, (k, n // 2), dtype=torch.uint8).cuda()
        )
        self.register_buffer(
            "scales",
            torch.randn(num_groups, n, dtype=torch.float16).cuda() * 0.01
        )
        self.register_buffer(
            "zeros",
            torch.zeros(num_groups, n, dtype=torch.float16).cuda() + 8.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [M, K]
        weight = dequantize_int4(self.qweight, self.scales, self.zeros)
        return x @ weight


def get_inputs():
    return [torch.randn(M, K, dtype=torch.float16).cuda()]


def get_init_inputs():
    return [K, N, GROUP_SIZE]
