# popcorn2: large-tier module centers (scripts/gen_popcorn2_centers.py).
# Source: KernelBench/level4/popcorn/28_stabilityai-stable-diffusion-xl-base-1.0.py

import torch
from diffusers import UNet2DConditionModel

class Model(torch.nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = UNet2DConditionModel.from_pretrained(self.model_name, subfolder='unet')

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.model(sample, timestep, encoder_hidden_states).sample
model_name = 'stabilityai/stable-diffusion-xl-base-1.0'
batch_size = 4
channels = 4
height = 64
width = 64
context_len = 77
cross_attn_dim = 2048

def get_inputs():
    p = popcorn_pri
    mode = p.sample_input_mode()
    sample = torch.randn(p.trial_dim(batch_size, 'batch_size', mode=mode), p.trial_dim(channels, 'channels', mode=mode, align=8), p.trial_dim(height, 'height', mode=mode), p.trial_dim(width, 'width', mode=mode))
    timestep = torch.randint(0, p.trial_dim(1000, '_', mode=mode), (p.trial_dim(batch_size, 'batch_size', mode=mode),), dtype=torch.long)
    encoder_hidden_states = torch.randn(p.trial_dim(batch_size, 'batch_size', mode=mode), p.trial_dim(context_len, 'context_len', mode=mode), p.trial_dim(cross_attn_dim, 'cross_attn_dim', mode=mode, align=8))
    return [sample, timestep, encoder_hidden_states]

def get_init_inputs():
    return [model_name]
