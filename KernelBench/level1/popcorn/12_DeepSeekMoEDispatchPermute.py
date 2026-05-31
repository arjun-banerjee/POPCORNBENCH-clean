import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Pack tokens into expert-major order for MoE execution.

    Each token is assigned to one expert and one slot within that expert's
    buffer. The kernel writes token activations into the packed dispatch layout.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        token_hidden: torch.Tensor,
        expert_idx: torch.Tensor,
        slot_idx: torch.Tensor,
        expert_offsets: torch.Tensor,
    ) -> torch.Tensor:
        num_rows = int(expert_offsets[-1].item())
        out = torch.zeros(num_rows, token_hidden.shape[1], dtype=token_hidden.dtype, device=token_hidden.device)
        for token in range(token_hidden.shape[0]):
            expert = int(expert_idx[token].item())
            row = int(expert_offsets[expert].item() + slot_idx[token].item())
            out[row] = token_hidden[token]
        return out


num_tokens = 2048
hidden_dim = 128
num_experts = 16


def get_inputs():
    return popcorn_pri.moe_dispatch_inputs(num_tokens, hidden_dim, num_experts)




def get_init_inputs():
    return []
