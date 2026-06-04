from math import gcd, ceil
import functools

import torch
from torch import nn, einsum
import torch.nn.functional as F

from Rotary_Embedding_torch import RotaryEmbedding, apply_rotary_emb

from einops import rearrange, repeat
import numpy as np


# -----------helper functions---------------
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class PerceiverARAttention(nn.Module):
    def __init__(
        self,
        *,
        dim = 512, # embedding size
        heads = 8,
        causal = True,
        sequence_len = 1024,
        latent_len = 256,  # history = sequence length - latent
        pos_emb = None,
        dropout = 0.,
        layer_num = 0
    ):
        super().__init__()
        self.sequence_len = sequence_len
        self.latent_len = latent_len
        self.history_len = self.sequence_len - self.latent_len
        assert (self.history_len + latent_len == sequence_len), 'history_length plus latent should be equal to sequence length'
        self.layer_num = layer_num # for PerceiverAR, first layer is different
        self.dim_head = dim//heads  # 512/8 = 64
        self.scale = self.dim_head ** -0.5  # 1/8

        self.heads = heads
        self.causal = causal

        self.norm = nn.LayerNorm(self.dim_head)  # (64)

        self.pos_emb = default(pos_emb, RotaryEmbedding(self.dim_head))
        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, dim, bias = False)  # e.g.,(512, 512)
        self.to_kv = nn.Linear(dim, dim, bias = False) # e.g., (512, 512)
        self.w_out = nn.Linear(dim, dim) # e.g., (512, 512)

    def forward(self, x, mask = None): # e.g., x has shape of (4,1024,512)
        b, n, *_, heads, device, causal = *x.shape, self.heads, x.device, self.causal, 
        # e.g., b = 4, n = 1024, *, h = 8, device, causal 

        mask_value = -torch.finfo(x.dtype).max # negative inf.

        # get queries, keys, values
        qkv = (self.to_q(x), self.to_kv(x))  # e.g., (4, 1024, 512)
        total_len = x.shape[-2]    # 1024
        # get sequence range, for calculating mask
        seq_range = torch.arange(total_len, device = device)

        # split heads
        q, kv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = heads), qkv)  # b = 4, n = 1024, h = 8, d = 64; 
        # e.g., q, kv = [32, 1024, 64]    
        if self.layer_num == 0:
            q = q[:,self.history_len:,:]
 
        # rotary embedding
        if exists(self.pos_emb):
            rotary_emb = self.pos_emb(seq_range, cache_key = total_len) # (1024, 64)
            rotary_emb = rearrange(rotary_emb, 'n d -> () n d') # (1,1024,64)
            if self.layer_num == 0:
                q = apply_rotary_emb(rotary_emb[:,self.history_len:,:],q)
                kv = apply_rotary_emb(rotary_emb,kv)

        # scale queries
        q = q * self.scale # (32,1024,64) - each head

        kv_norm = self.norm(kv)
        attn = einsum('b i d, b j d -> b i j', q, kv_norm)  
        # [32, 256, 1024]; b = 32, (attention of latent_lenxsequence_len e.g., 256x1024)

        m_size = attn.shape[-2]  # size of mask

        if exists(mask): # called when generate triggers
            mask = rearrange(mask, 'b n -> b () n') # [8, 8, 1, 256]
            attn.masked_fill_(~mask, mask_value)

        if self.causal:
            if self.layer_num == 0: # last q part of attention is masked in first layer 
                causal_mask = torch.ones(m_size, m_size, device = device).triu_(1).bool()
                attn[:,:,self.history_len:].masked_fill_(causal_mask, mask_value)
            else:
                causal_mask = torch.ones(m_size, m_size, device = device).triu_(1).bool()
                attn.masked_fill_(causal_mask, mask_value)
 
        
        # final attention
        attn = attn.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values (same as keys, since tied) and project out
        out = einsum('b i j, b j d -> b i d', attn, kv_norm)
        out = rearrange(out, '(b h) n d -> b (n) (h d)', h = heads)
        #out = out[:, :n]
        return self.w_out(out)


class PerceiverARTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,  # for output size determination
        dim,   # embedding
        latent_len,
        num_layers, # number of layers
        heads = 8,
        sequence_len,
        causal = True,
        ff_mult = 4,  # expand feedforward by 4 times then reduce back to embedding size
        ff_dropout = 0.,
        attn_dropout = 0.
    ):
        super().__init__()
        self.sequence_len = sequence_len
        self.latent_len = latent_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.dim_head = dim//heads
        pos_emb = RotaryEmbedding(self.dim_head)
 
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, PerceiverARAttention(dim = dim, heads = heads, sequence_len = sequence_len, latent_len = latent_len,causal = causal,layer_num=i, pos_emb = pos_emb, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, mask = None):
        x = self.token_emb(x)
        for attn, ff in self.layers:
            if attn.fn.layer_num == 0:  # layer_num
                x = attn(x, mask = mask) + x[:,-self.latent_len:,:]
            else:
                x = attn(x, mask = mask) + x
            x = ff(x) + x   #  [4, 1024, 512]
        return self.to_logits(x)
