import torch
import math, dataclasses, inspect, rich, os 
import torch.nn as nn 
import torch.nn.functional as func_nn 

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