"""
Fused RMS LayerNorm.

Computes: out = x / rms(x) * weight   where rms(x) = sqrt(mean(x^2) + eps).
The 'fused' variant fuses an optional residual add with the norm.
"""
import torch
import torch.nn as nn

HIDDEN_SIZE = 4096
NUM_TOKENS = 512
EPS = 1e-5


class Model(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [num_tokens, hidden_size]
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return (x_norm * self.weight).to(x.dtype)


def get_inputs():
    x = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.float16).cuda()
    return [x]


def get_init_inputs():
    return [HIDDEN_SIZE, EPS]
