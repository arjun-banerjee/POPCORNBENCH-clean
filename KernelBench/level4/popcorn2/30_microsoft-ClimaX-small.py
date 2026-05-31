# popcorn2: large-tier module centers (scripts/gen_popcorn2_centers.py).
# Source: KernelBench/level4/popcorn/30_microsoft-ClimaX-small.py

import torch
from transformers import AutoConfig, AutoModel

class Model(torch.nn.Module):

    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModel.from_pretrained(self.model_name, config=self.config)

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        if hasattr(out, 'last_hidden_state'):
            return out.last_hidden_state
        return out[0]
model_name = 'microsoft/ClimaX-small'
config = AutoConfig.from_pretrained(model_name)
batch_size = 4
channels = 3
height = 224
width = 224

def get_inputs():
    p = popcorn_pri
    mode = p.sample_input_mode()
    return [torch.randn(p.trial_dim(batch_size, 'batch_size', mode=mode), channels, p.trial_dim(height, 'height', mode=mode), p.trial_dim(width, 'width', mode=mode))]

def get_init_inputs():
    return [model_name, config]
