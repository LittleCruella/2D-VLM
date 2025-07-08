from dataclasses import dataclass, field
from typing import Optional
import os
import torch
import torch.distributed as dist
import transformers
from safetensors.torch import load_file
from transformers import AutoTokenizer




def is_rank_zero():
    if "RANK" in os.environ:
        if int(os.environ["RANK"]) != 0:
            return False
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return False
    return True

print(f"Is rank zero: {is_rank_zero()}")