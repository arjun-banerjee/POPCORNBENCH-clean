# popcorn2: large-tier module centers (scripts/gen_popcorn2_centers.py).
# Source: KernelBench/level2/popcorn/50_HiddenMarkovForward.py

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Forward algorithm for Hidden Markov Models.  Computes the log-
    likelihood log p(observations) by iterating the forward recursion
    in log-space for numerical stability.  The dominant inner loop in
    HMM training (Baum-Welch) and Viterbi decoding.
    """

    def __init__(self, num_states, num_obs):
        super().__init__()
        self.num_states = num_states
        self.num_obs = num_obs
        self.log_init = nn.Parameter(torch.zeros(num_states))
        self.log_trans = nn.Parameter(torch.zeros(num_states, num_states))
        self.log_emit = nn.Parameter(torch.zeros(num_states, num_obs))

    def _normalize_log(self, x, dim=-1):
        return x - torch.logsumexp(x, dim=dim, keepdim=True)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (B, T) – integer observation indices (as long tensor)
        Returns:
            log_lik: (B,) – log p(observations) under the HMM
        """
        (B, T) = observations.shape
        log_pi = self._normalize_log(self.log_init)
        log_A = self._normalize_log(self.log_trans, dim=-1)
        log_B = self._normalize_log(self.log_emit, dim=-1)
        emit_0 = log_B[:, observations[:, 0]].t()
        log_alpha = log_pi.unsqueeze(0) + emit_0
        for t in range(1, T):
            log_alpha = log_alpha.unsqueeze(-1) + log_A.unsqueeze(0)
            log_alpha = torch.logsumexp(log_alpha, dim=1)
            emit_t = log_B[:, observations[:, t]].t()
            log_alpha = log_alpha + emit_t
        log_lik = torch.logsumexp(log_alpha, dim=-1)
        return log_lik
num_states = 16
num_obs = 64
seq_len = 256
batch_size = 64

def get_inputs():
    p = popcorn_pri
    mode = p.sample_input_mode()
    return [torch.randint(0, num_obs, (p.trial_dim(batch_size, 'batch_size', mode=mode), p.trial_dim(seq_len, 'seq_len', mode=mode)))]

def get_init_inputs():
    return [num_states, num_obs]
