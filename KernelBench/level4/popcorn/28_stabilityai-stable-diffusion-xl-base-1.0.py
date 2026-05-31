import torch
from diffusers import UNet2DConditionModel


class Model(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = UNet2DConditionModel.from_pretrained(self.model_name, subfolder="unet")

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.model(sample, timestep, encoder_hidden_states).sample


model_name = "stabilityai/stable-diffusion-xl-base-1.0"
batch_size = 2
channels = 4
height = 64
width = 64
context_len = 77
cross_attn_dim = 2048


def get_inputs():
    p = popcorn_pri
    sample = torch.randn(p.jitter_int(batch_size), p.jitter_int(channels, align=8), p.jitter_int(height), p.jitter_int(width))
    timestep = torch.randint(0, p.jitter_int(1000) , (p.jitter_int(batch_size),), dtype=torch.long)
    encoder_hidden_states = torch.randn(p.jitter_int(batch_size), p.jitter_int(context_len), p.jitter_int(cross_attn_dim, align=8))
    return [sample, timestep, encoder_hidden_states]


def get_init_inputs():
    return [model_name]
