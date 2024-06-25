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
k_heads = 12
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

ptdtype={'float32': torch.float32, 'bfloat16': torch.bfloat16,
    'float16': torch.float16}[dtype]

ctx=nullcontext() if device_type == 'cpu' else torch.amp.autocast(
    device_type=device_type, dtype=ptdtype)

# poor man's dataloader
data_dir=os.path.join('data', dataset)

def get_batch(split):
    if split == 'train':
        data=np.memmap(os.path.join(data_dir, 'train.bin'),
                       dtype=np.uint16, mode='r')
    else:
        data=np.memmap(os.path.join(data_dir, 'val.bin'),
                       dtype=np.uint16, mode='r')

    ix=torch.randint(len(data) - block_size, (batch_size,))

    x=torch.stack(
        [torch.from_numpy((data[k: k + block_size]).astype(np.int64)) for k in ix])
    y=torch.stack(
        [torch.from_numpy((data[k+1: k+1+block_size]).astype(np.int64)) for k in ix])

    if device_type == 'cuda':
        x, y=x.pin_memory().to(device, non_blocking=True),  y.pin_memory().to(
            device, non_blocking=True)
    else:
        x, y=x.to(device), y.to(device)

    return x, y


iter_num=0
best_val_loss=1e9


meta_path=os.path.join(data_dir, 'meta.pkl')
meta_vocab_size=None

if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta=pickle.load(f)
    meta_vocab_size=meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args=dict(n_layer=n_layer, k_heads=k_heads, embed_dim=embed_dim,
                block_size=block_size, bias=bias, vocab_size=vocab_size, dropout=dropout)  # model args

if init_from == 'scratch':
    print('initializing a new model from scratch')
    if meta_vocab_size is None:
        print('default to gpt2 vocab size 50304/50257')
    model_args['vocab_size']=meta_vocab_size if meta_vocab_size is not None else 50304

    gptconf=GPTconfig(**model_args)
    model=MiniGPT(gptconf)

elif init_from == 'resume':
    print(f'resuming training from {out_dir}')
    # continue training from checkpoint
    ckpt_path=os.path.join(out_dir, 'mini_gpt.pt')
    checkpoint=torch.load(ckpt_path, map_location=device)
    checkpoint_model_args=checkpoint['model_args']

    for k in ['n_layer', 'k_heads', 'embed-dim', 'block_size', 'bias', 'vocab_size']:
        model_args[k]=checkpoint_model_args[k]

    # init model
    gpt_conf=GPTconfig(**model_args)
    model=MiniGPT(gpt_conf)
    state_dict=checkpoint['model']

    unwanted_prefix='_orig_mod.'

    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]]=state_dict.pop(k)

    model.load_state_dict(state_dict)
    iter_num=checkpoint['iter_num']
    best_val_loss=checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
