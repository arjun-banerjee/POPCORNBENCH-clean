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


model_name = "HuggingFaceM4/idefics2-8b"
config = AutoConfig.from_pretrained(model_name)
vocab_size = config.vocab_size
sequence_length = 128
batch_size = 2
image_size = 224


def get_inputs():
    p = popcorn_pri
    bs = p.jitter_int(batch_size)
    sl = p.jitter_int(sequence_length)
    inputs = torch.randint(0, vocab_size, (bs, sl))
    return [inputs]



def get_init_inputs():
    return [model_name, config]
