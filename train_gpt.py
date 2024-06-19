import os, time, math, pickle
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from torch.distributed import init_process_group, destroy_process_group
from mini_gpt import GPTconfig, MiniGPT

# default config dfor training
out_dir = 'mini_gpt'
eval_interval = 2000
log_interval = 1
eval_only = False
eval_iters = 200
init_from = 'scratch'

# wandb 
wandblog = False
wandb_project = 'mini_gpt'
wandb_run = 'research_gpt_1'

dataset = 'openwebtext'

