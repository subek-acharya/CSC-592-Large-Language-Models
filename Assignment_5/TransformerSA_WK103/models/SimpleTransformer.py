from torch import nn
from .TransformerBlock import TransformerBlock
from .PositionalEncoding import PositionalEncoding
import torch

class SimpleTransformer(nn.Module):
   def __init__(self, dim, num_unique_tokens=256, num_layers=6, heads=8, dim_head=None, max_seq_len=1024, causal=True):
       super().__init__()
       self.max_seq_len = max_seq_len
       self.causal=causal
       self.token_emb = nn.Embedding(num_unique_tokens, dim)

       # our position embedding class
       self.pos_enc = PositionalEncoding(dim,max_seq_length=max_seq_len)
              
       self.block_list = [TransformerBlock(dim=dim, heads=heads, dim_head=dim_head,causal=causal) for _ in range(num_layers)]
       self.layers = nn.ModuleList(self.block_list)

       self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_unique_tokens)
        )

   def set_causal(self, causal):
       for b in self.block_list:
           b.set_causal(causal)

   def forward(self, x, mask=None):
       
       x = self.token_emb(x)
       x = x + self.pos_enc(x)
       for layer in self.layers:
           x = layer(x, mask)
       return self.to_logits(x)
