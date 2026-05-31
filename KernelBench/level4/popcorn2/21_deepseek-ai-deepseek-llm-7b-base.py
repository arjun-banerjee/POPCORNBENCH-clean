# popcorn2: large-tier module centers (scripts/gen_popcorn2_centers.py).
# Source: KernelBench/level4/popcorn/21_deepseek-ai-deepseek-llm-7b-base.py

import torch
from transformers import AutoConfig, AutoModelForCausalLM

class Model(torch.nn.Module):

    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)

    def forward(self, x):
        return self.model(x).logits
model_name = 'deepseek-ai/deepseek-llm-7b-base'
config = AutoConfig.from_pretrained(model_name)
vocab_size = config.vocab_size
sequence_length = 1024
batch_size = 8

def get_inputs():
    p = popcorn_pri
    mode = p.sample_input_mode()
    bs = p.trial_dim(batch_size, 'batch_size', mode=mode)
    sl = p.trial_dim(sequence_length, 'sequence_length', mode=mode)
    inputs = torch.randint(0, vocab_size, (bs, sl))
    return [inputs]

def get_init_inputs():
    return [model_name, config]
