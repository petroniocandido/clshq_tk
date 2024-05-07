import torch
from torch import nn

from clshq_tk.modules.attention import MultiHeadAttention

class Transformer(nn.Module):
  def __init__(self, num_heads, num_tokens,  embed_dim, feed_forward, 
               device = None, dtype=torch.float64, **kwargs):
    super().__init__()

    self.num_heads = num_heads
    self.num_tokens = num_tokens
    self.embed_dim = embed_dim
    self.device = device
    self.dtype = dtype
    self.attention = MultiHeadAttention(num_heads, num_tokens, embed_dim, 
                            dtype=self.dtype, device=self.device)
    self.ln = nn.LayerNorm(embed_dim)
    self.flat = nn.Flatten(1)    
    self.linear1 = nn.Linear(num_tokens * embed_dim, feed_forward, 
                            dtype=self.dtype, device=self.device)
    self.linear2 = nn.Linear(feed_forward, num_tokens * embed_dim, 
                            dtype=self.dtype, device=self.device)
    self.gelu = nn.GELU()
    self.drop = nn.Dropout(.25)
    self.unflat = nn.Unflatten(1, [num_tokens, embed_dim])

  def forward(self, x):
    z = self.attention(x)
    z = self.ln(x + z)
    z = self.flat(z)
    z = self.gelu(self.linear1(self.drop(z)))
    z = self.gelu(self.linear2(self.drop(z)))
    z = self.unflat(z)
    z = self.ln(x + z)
    return z

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.attention = self.attention.to(*args, **kwargs)
    self.linear1 = self.linear1.to(*args, **kwargs)
    self.linear2 = self.linear2.to(*args, **kwargs)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    self.attention = self.attention.train(*args, **kwargs)
    self.linear1 = self.linear1.train(*args, **kwargs)
    self.linear2 = self.linear2.train(*args, **kwargs)
    return self
