import os
import time
import math
import pickle
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from torch.distributed import init_process_group, destroy_process_group
from mini_gpt import GPTconfig, MiniGPT

# default config dfor training
out_dir = "mini_gpt"
eval_interval = 200
log_interval = 1
eval_only = False
eval_iters = 200
init_from = "scratch"
# wandb
wandblog = False
wandb_project = "mini_gpt"
wandb_run = "research_gpt_1"

# model hparams
dataset = "openwebtext"
grad_acc_steps = 5 * 8
batch_size = 12
block_size = 1024
n_layer = 12
n_head = 12
dropout = 0.0  # 0 for base, 0.1+ for finetuning
bias = False

# optimizer
lr = 6e-4
weight_decay = 1e-1
beta_1 = 0.9
beta_2 = 0.95
grad_clip = 1.0

# learning rate/decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
# settings
backend = "nccl"
dtype = "bfloat16"
device = (
    "cuda"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = True


config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]

exec(open('configurator.py').read())  # overrides from command line
config = {k: globals()[k] for k in config_keys}  # for logging

# various inits deriv attributes, i/o setup
# config for distributed training
ddp = init(os.environ.get('RANK', -1) != -1

if ddp:
    init_process_group(backend=backend)
    ddp_rank=int(os.environ['RANK'])
    ddp_local_rank=int(os.environ['RANK']
    device=f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    # this process will do logging, checkpointing etc.
    master_process=ddp_rank == 0
    seed_offset=ddp_rank  # each process gets a different seed
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process=True
    seed_offset=0
    ddp_world_size=1

tokens_per_iter=gradient_accumulation_steps * \
    ddp_world_size * batch_size * block_size
print(f'n_tokens per step => {tokens_per_iter:,}')
if master_process:
    os.mkdir(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
device_type='cuda' if 'cuda' in device else 'cpu'  # for torch.autocast
