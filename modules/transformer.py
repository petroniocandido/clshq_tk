import torch
from torch import nn

from clshq_tk.modules.attention import MultiHeadAttention

class Transformer(nn.Module):
  def __init__(self, num_heads, num_tokens,  embed_dim, feed_forward, device = None, **kwargs):
    super().__init__()

    self.num_heads = num_heads
    self.num_tokens = num_tokens
    self.embed_dim = embed_dim
    self.device = device
    self.attention = MultiHeadAttention(num_heads, num_tokens, embed_dim)
    self.ln = nn.LayerNorm(embed_dim)
    self.flat = nn.Flatten(1)    
    self.linear1 = nn.Linear(num_tokens * embed_dim, feed_forward)
    self.linear2 = nn.Linear(feed_forward, num_tokens * embed_dim)
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
    self.device = args[0]
    self.attention = self.attention.to(self.device)
    self.linear1 = self.linear1.to(self.device)
    self.linear2 = self.linear2.to(self.device)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    self.attention = self.attention.train(*args, **kwargs)
    self.linear1 = self.linear1.train(*args, **kwargs)
    self.linear2 = self.linear2.train(*args, **kwargs)
    return self
