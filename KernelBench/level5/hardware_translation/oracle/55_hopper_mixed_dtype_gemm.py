"""
Hopper mixed-dtype GEMM.

Source A100 kernel: 08_turing_tensorop_gemm.
Target H100 kernel: 55_hopper_mixed_dtype_gemm.

Reference: BF16 activations multiplied by a grouped INT8 weight that is
dequantized to float for the matmul.
"""
import torch
import torch.nn as nn

M = 2048
N = 2048
K = 2048
GROUP_SIZE = 128


class Model(nn.Module):
    def __init__(self, k: int, n: int, group_size: int):
        super().__init__()
        groups = (k + group_size - 1) // group_size
        self.group_size = group_size
        self.register_buffer("qweight", torch.randint(-8, 8, (k, n), dtype=torch.int8).cuda())
        self.register_buffer("scale", (torch.rand(groups, n, dtype=torch.float32).cuda() * 0.05) + 0.01)
        self.register_buffer("zero", torch.zeros(groups, n, dtype=torch.float32).cuda())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        groups = self.scale.shape[0]
        padded_k = groups * self.group_size
        qweight = self.qweight
        if qweight.shape[0] != padded_k:
            pad = padded_k - qweight.shape[0]
            qweight = torch.nn.functional.pad(qweight, (0, 0, 0, pad))
        weight = qweight.view(groups, self.group_size, -1).to(torch.float32)
        weight = (weight - self.zero.unsqueeze(1)) * self.scale.unsqueeze(1)
        weight = weight.view(padded_k, -1)[: self.qweight.shape[0]]
        return (x.to(torch.float32) @ weight).to(x.dtype)


def get_inputs():
    return [torch.randn(M, K, dtype=torch.bfloat16).cuda()]


def get_init_inputs():
    return [K, N, GROUP_SIZE]

