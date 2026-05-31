# popcorn2: large-tier module centers (scripts/gen_popcorn2_centers.py).
# Source: KernelBench/level4/popcorn/26_smolvla-SmolVLA-Base.py

import torch
from transformers import AutoConfig, AutoModelForVision2Seq

class Model(torch.nn.Module):

    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name, config=self.config)

    def forward(self, input_ids, pixel_values):
        return self.model(input_ids=input_ids, pixel_values=pixel_values).logits
model_name = 'smolvla/SmolVLA-Base'
config = AutoConfig.from_pretrained(model_name)
vocab_size = config.vocab_size
sequence_length = 256
batch_size = 1
image_size = 224

def get_inputs():
    p = popcorn_pri
    mode = p.sample_input_mode()
    bs = p.trial_dim(batch_size, 'batch_size', mode=mode)
    sl = p.trial_dim(sequence_length, 'sequence_length', mode=mode)
    inputs = torch.randint(0, vocab_size, (bs, sl))
    return [inputs]

def get_init_inputs():
    return [model_name, config]
