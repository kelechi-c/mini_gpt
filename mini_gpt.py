from turtle import st
import torch
import math, dataclasses, inspect, rich, os 
import torch.nn as nn 
import torch.nn.functional as func_nn 
from dataclasses import dataclass 
class LayerNorm(nn.Module): # Layer normalization 
    def __init__(self, k_dims, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(k_dims))
        self.bias = nn.Parameter(torch.zeros(k_dims)) if bias else None
        
    def forward(self, input):
        output = func_nn.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        
        return output 
    
class SelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.n_heads == 0
        self.attention = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=config.bias) # query key value projections for attention heads in a batch
        self.project = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.dropout) # regularization
        self.residual_drop = nn.Dropout(config.dropout)
        self.k_heads = config.k_heads
        self.embed_dim = config.embed_dim
        self.dropout = config.dropout
        
        # flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_poduct_attention')
        if not self.flash:
            rich.print('[bold red] not using flash attention!')
            self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    
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
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            
        else:
            attention = (q @ k.transpose(-2, -1)) * (1.0 /math.sqrt(k.size(-1)))
            attention = attention.masked_fill(self.bias[:,:,:C,:C] == 0, float('-inf'))
            attention = func_nn.softmax(attention, dim=1)
            attention = self.attention_dropout(attention)
            y = attention @ v 
            
        y = y.transpose(1, 2).contiguous().view(A, B, C) # reassemble all to be side by side
        
        # output projection
        attn_output = self.residual_drop(self.project(y))
        
        return attn_output
    
class MultiLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.embed_dim, 4*config.embed_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.project = nn.Linear(4*config.embed_dim, config.embed_dim, bias=config.bias)
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
        
        self.transformer = nn.ModuleDict(dict(
            word_te = nn.Embedding(config.vocab_size, config.embed_dim),
            word_pe = nn.Embedding(config.block_size, config.embed_dim),
            drop = nn.Dropout(config.dropout),
            attention_layer = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            layer_norm_f = LayerNorm(config.embed_dim, bias=config.bias)     
        )) 
        
        self.linear_head = nn.Linear(config.embed_dim, config.vocab_size, bias=config.bias)
        self.transformer.word_te.weight = self.linear_head.weight
        
        self.apply(self._init_weights) # init weights for model
        
        for pn, p in self.named_parameters():
            if pn.endswith('project.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.002/math.sqrt(2 * config.n_layers))
                
        # display number of parameters in model
        print(f'number of parameters => {self.get_num_params()/1e6:.2f}')
        
    def get_num_params(self, non_embedding=True): #returns the number of parameters in the model
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
     