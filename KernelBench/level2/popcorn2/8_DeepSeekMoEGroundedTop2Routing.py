# popcorn2: large-tier module centers (scripts/gen_popcorn2_centers.py).
# Source: KernelBench/level2/popcorn/8_DeepSeekMoEGroundedTop2Routing.py

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Grounded top-2 MoE router.

    Router logits are biased by token-expert similarity against learned
    grounding embeddings, then normalized over the selected top-2 experts.
    """

    def __init__(self):
        super().__init__()

    def forward(self, token_hidden: torch.Tensor, router_logits: torch.Tensor, expert_ground: torch.Tensor, alpha: float) -> torch.Tensor:
        grounded = router_logits + alpha * (token_hidden @ expert_ground.t())
        (top_vals, top_idx) = torch.topk(grounded, k=2, dim=-1)
        top_weights = torch.softmax(top_vals, dim=-1)
        return torch.stack((top_idx.to(torch.float32), top_weights), dim=-1)
num_tokens = 8192
hidden_dim = 160
num_experts = 16
alpha = 0.35

def get_inputs():
    p = popcorn_pri
    mode = p.sample_input_mode()
    nt = p.trial_dim(num_tokens, 'num_tokens', mode=mode)
    hd = p.trial_dim(hidden_dim, 'hidden_dim', mode=mode, align=16)
    ne = p.trial_dim(num_experts, 'num_experts', mode=mode, minimum=2)
    token_hidden = torch.randn(nt, hd, dtype=torch.float32)
    router_logits = torch.randn(nt, ne, dtype=torch.float32)
    expert_ground = torch.randn(ne, hd, dtype=torch.float32)
    return [token_hidden, router_logits, expert_ground, alpha]

def get_init_inputs():
    return []
