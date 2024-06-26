from turtle import st
import torch
import math
import dataclasses
import inspect
import rich
import os
import torch.nn as nn
import torch.nn.functional as func_nn
from dataclasses import dataclass


class LayerNorm(nn.Module):  # Layer normalization
    def __init__(self, k_dims, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(k_dims))
        self.bias = nn.Parameter(torch.zeros(k_dims)) if bias else None

    def forward(self, input):
        output = func_nn.layer_norm(
            input, self.weight.shape, self.weight, self.bias, 1e-5
        )

        return output


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.n_heads == 0
        # query key value projections for attention heads in a batch
        self.attention = nn.Linear(
            config.embed_dim, 3 * config.embed_dim, bias=config.bias
        )
        self.project = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.dropout)  # regularization
        self.residual_drop = nn.Dropout(config.dropout)
        self.k_heads = config.k_heads
        self.embed_dim = config.embed_dim
        self.dropout = config.dropout

        # flash attention
        self.flash = hasattr(torch.nn.functional, "scaled_dot_poduct_attention")
        if not self.flash:
            rich.print("[bold red] not using flash attention!")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, input):
        # batch_size, sequence_length, embedding_dims
        A, B, C = input.size()

        # calc for query-key-value for all attention heads in batch and move batch forward to batch dim
        q, k, v = self.attention(input).split(self.embed_dim, dim=2)

        # view and transpose leading to yield shape of (A, k_heads, B, hs)
        q = q.view(A, B, self.k_heads, C // self.k_heads).transpose(1, 2)
        k = k.view(A, B, self.k_heads, C // self.k_heads).transpose(1, 2)
        v = v.view(A, B, self.k_heads, C // self.k_heads).transpose(1, 2)

        # causal self attention (A, kheads, B, hs) x (A, kheads, hs, C) -> (A, kheads, C, C)
        if self.flash:
            # implementation of flash attention using cuda kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )

        else:
            attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attention = attention.masked_fill(
                self.bias[:, :, :C, :C] == 0, float("-inf")
            )
            attention = func_nn.softmax(attention, dim=1)
            attention = self.attention_dropout(attention)
            y = attention @ v

        y = (
            y.transpose(1, 2).contiguous().view(A, B, C)
        )  # reassemble all to be side by side

        # output projection
        attn_output = self.residual_drop(self.project(y))

        return attn_output


class MultiLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.project = nn.Linear(
            4 * config.embed_dim, config.embed_dim, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        x = self.gelu(self.fc(input))
        output = self.dropout(self.project(x))

        return output


class Block(nn.Module):
    def __init__(self, config):
        self.layer_norm_1 = LayerNorm(config.embed_dim, bias=config.bias)
        self.attention = SelfAttention(config)
        self.layer_norm_2 = LayerNorm(config.embed_dim, bias=config.bias)
        self.mlp = MultiLP(config)

    def forward(self, input):
        input = input + self.attention(self.layer_norm_1(input))
        output = input + self.mlp(self.layer_norm_2(input))

        return output


@dataclass
class GPTconfig:
    block_size = 1024
    vocab_size = 50304
    n_layers = 12
    k_heads = 12
    embed_dim = 768
    dropout = 0.0
    bias = True


class MiniGPT(nn.Module):
    def __init__(self, config: GPTconfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                word_te=nn.Embedding(config.vocab_size, config.embed_dim),
                word_pe=nn.Embedding(config.block_size, config.embed_dim),
                drop=nn.Dropout(config.dropout),
                attention_layer=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layers)]
                ),
                layer_norm_f=LayerNorm(config.embed_dim, bias=config.bias),
            )
        )

        self.linear_head = nn.Linear(
            config.embed_dim, config.vocab_size, bias=config.bias
        )
        self.transformer.word_te.weight = self.linear_head.weight

        self.apply(self._init_weights)  # init weights for model

        for pn, p in self.named_parameters():
            if pn.endswith("project.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.002 / math.sqrt(2 * config.n_layers)
                )
        # display number of parameters in model
        print(f"number of parameters => {self.get_num_params()/1e6:.2f}")

    # returns the number of parameters in the model
    def get_num_params(self, non_embedding=True):
        num_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            num_params -= self.transformer.word_pe.weight.numel()

        return num_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.wight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"cannot forward sequence of len{t}, block size is {self.config.block_size}"
        position = torch.arange(0, t, dtype=torch.long, device=device)

        # forward pass for gpt
        tokenembed = self.transformer.word_te(idx)
        posembed = self.transformer.word_pe(position)

        x = self.transformer.drop(tokenembed + posembed)

        for block in self.transformer.attention_layer:
            x = block(x)

        x = self.transformer.layer_norm_f(x)

        if targets is not None:
            logits = self.linear_head(x)
            # calc the loss if given desired targets
            loss = func_nn.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference optimization to forward only the last position
            logits = self.linear_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # for block size interchange when loading pretrained model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.word_pe.weight = nn.Parameter(
            self.transformer.word_pe.weight[:block_size]
        )
        for block in self.transformer.attention_layer:
            if hasattr(block.attention, "bias"):
                block.attention.bias = block.attention.bias[:, :, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default tp empty
        assert all(k == "dropout" for k in override_args)

        from transformers import GPT2LMHeadModel

        print(f"Loading weights from pretrained gpt2 => {model_type}")

        # n_layers, k_heads, and embed_dim
        config_args = {
            # 124M params
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            # 350M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            # 774M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            # 1558M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        print(f"forcing vocab_size=50257, block_size=1024, bias=True")
        # the same for gpt model checkpoints
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True

        # you can override dropout
        if "dropout" in override_args:
            print(f"overriding dropout to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]

        # create initalized gpt model from scratch
        config = GPTconfig(**config_args)
        gpt_model = MiniGPT(config)
        state = gpt_model.state_dict()
        state_keys = state.keys()
        state_keys = [k for k in state_keys if not k.endswith(".attention.bias")]

        # load hf transformers model
        hf_gpt = GPT2LMHeadModel.from_pretrained(model_type)
        state_hf = hf_gpt.state_dict()

        # copy in alignment
        state_keys_hf = state_hf.keys()
        state_keys_hf = [
            k for k in state_keys_hf if not k.endswith(".attention.maskedbias")
        ]
        state_keys_hf = [k for k in state_keys_hf if not k.endswith(".attention.bias")]

        transposed = [
            "attention.attention.weight",
            "attention.project.weight",
            "mlp.fc.weight",
            "mlp.project.weight",
        ]
        # openai chekpoints use a conv1D module
        # transpose to vanilla linear weights

        assert (
            len(state_keys_hf) == len(state_keys)
        ), f"mismatch of state keys {len(state_keys_hf)} for hf and {len(state_keys)} for scratch"
        for k in state_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # solution for conv1d weights
                assert state_hf[k].shape[::-1] == state[k].shape
                with torch.no_grad():
                    state[k].copy(state_hf[k].t())
            else:
                assert state_hf[k].shape == state[k].shape
                with torch.no_grad():
                    state[k].copy_(state_hf[k])

        return gpt_model

    def configure_optimizers(self, weight_decay, learn_rate, betas, device):
        # start with * parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those thta require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim grous
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in no_decay_params)

        print(
            f"N of decayed param tensors => {len(decay_params)} with {num_decay_params,} parameters"
        )
        print(
            f"N of decayed param tensors => {len(no_decay_params)} with {num_nodecay_params,} parameters"
        )

        # init AdamW optimizer,
        fused_avail = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avail and device == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learn_rate, betas=betas, **extra_args
        )
        print(f"Using fused AdamW optimizer: {use_fused}")

        return optimizer

    def estimate_mfu(self, backforth_per_step, dt):  # estimate model flops utilization
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = (
            cfg.n_layers,
            cfg.k_heads,
            cfg.embed_dim // cfg.k_heads,
            cfg.block_size,
        )

        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_backforth = flops_per_token * T
        flops_per_iter = flops_per_backforth * backforth_per_step
        # express flops as ratio with A16 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # peak flops for A100

        mfu = flops_achieved / flops_promised  # get the ratio

        return mfu

    @torch.no_grad
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # crop the sequence context at block_size if its growing too log
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            logits, _ = self(idx_cond)  # forward pass
            # scale by desired temperature
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                # crop the logits to only the top_k options...optionally
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))

            # apply softmax for normalized distro
            probs = func_nn.softmax(logits, dim=1)
            # sample from the distribution
            next_idx = torch.multinomial(probs, num_samples=1)
            out_idx = torch.cat((idx, next_idx), dim=1)

        return out_idx
