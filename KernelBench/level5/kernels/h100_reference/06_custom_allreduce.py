"""
Custom AllReduce: SUM all-reduce across distributed ranks.

Multi-GPU: reduces the tensor across all ranks in the default process group.
Single-GPU: identity (no communication needed).
"""
import torch
import torch.distributed as dist
import torch.nn as nn

from kernelbench.distributed_collectives import (
    default_device,
    get_world_size,
    is_distributed_run,
    maybe_init_process_group,
)

HIDDEN = 4096
NUM_TOKENS = 512


class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maybe_init_process_group()
        if not is_distributed_run():
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x / get_world_size()


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(42)
    return [torch.randn(NUM_TOKENS, HIDDEN, device=dev, dtype=torch.float16,
                        generator=g)]


def get_init_inputs():
    return []
