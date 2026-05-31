# popcorn2: large-tier module centers (scripts/gen_popcorn2_centers.py).
# Source: KernelBench/level2/popcorn/6_TriangularAttention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Triangular self-attention over pair representations, as used in
    AlphaFold2's Evoformer stack.  Starting-node variant: for each
    row i of the pair matrix, standard multi-head self-attention is
    performed across the column dimension j, with an additive pair
    bias.  This lets positions (i,j) and (i,k) exchange information
    through their shared starting node i.
    """

    def __init__(self, pair_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = pair_dim // num_heads
        assert pair_dim % num_heads == 0
        self.layer_norm = nn.LayerNorm(pair_dim)
        self.query = nn.Linear(pair_dim, pair_dim, bias=False)
        self.key = nn.Linear(pair_dim, pair_dim, bias=False)
        self.value = nn.Linear(pair_dim, pair_dim, bias=False)
        self.gate = nn.Linear(pair_dim, pair_dim)
        self.bias_proj = nn.Linear(pair_dim, num_heads, bias=False)
        self.out_proj = nn.Linear(pair_dim, pair_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pair_repr: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_repr: (B, N, N, pair_dim)
            pair_mask: (B, N, N)
        Returns:
            updated pair representation (B, N, N, pair_dim)
        """
        (B, N, _, D) = pair_repr.shape
        h = self.num_heads
        d = self.head_dim
        x = self.layer_norm(pair_repr)
        g = torch.sigmoid(self.gate(x))
        bias = self.bias_proj(x).permute(0, 3, 1, 2)
        x_flat = x.reshape(B * N, N, D)
        q = self.query(x_flat).view(B * N, N, h, d).transpose(1, 2)
        k = self.key(x_flat).view(B * N, N, h, d).transpose(1, 2)
        v = self.value(x_flat).view(B * N, N, h, d).transpose(1, 2)
        attn = q @ k.transpose(-1, -2) / math.sqrt(d)
        attn = attn.view(B, N, h, N, N)
        attn = attn + bias.unsqueeze(1)
        mask = pair_mask.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.view(B * N, h, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, N, D)
        out = g * out
        return self.out_proj(out)
pair_dim = 64
num_heads = 4
seq_len = 64
batch_size = 4

def get_inputs():
    p = popcorn_pri
    mode = p.sample_input_mode()
    pair_repr = torch.randn(p.trial_dim(batch_size, 'batch_size', mode=mode), p.trial_dim(seq_len, 'seq_len', mode=mode), p.trial_dim(seq_len, 'seq_len', mode=mode), pair_dim)
    pair_mask = torch.ones(p.trial_dim(batch_size, 'batch_size', mode=mode), p.trial_dim(seq_len, 'seq_len', mode=mode), p.trial_dim(seq_len, 'seq_len', mode=mode))
    return [pair_repr, pair_mask]

def get_init_inputs():
    return [pair_dim, num_heads]
