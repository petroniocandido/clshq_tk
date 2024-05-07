
import torch
from torch import nn

from clshq_tk.modules.positional import PositionalEncoder
from clshq_tk.modules.transformer import Transformer

class TSAttentionClassifier(nn.Module):
  def __init__(self, tokenizer, num_tokens, num_layers, num_heads, feed_forward, 
               device = None, dtype = torch.float64, **kwargs):
    super().__init__()

    self.num_tokens = num_tokens
    self.num_layers = num_layers
    self.tokenizer = tokenizer
    self.tokenizer.freeze()
    self.device = device
    self.dtype = dtype
    self.positional = PositionalEncoder(self.num_tokens, self.tokenizer.embed_dim, 
                            dtype=self.dtype, device=self.device)
    self.transformers = [Transformer(num_heads, self.num_tokens, self.tokenizer.embed_dim, feed_forward, 
                         dtype=self.dtype, device=self.device) 
                         for k in range(num_layers)]
    self.flat = nn.Flatten(1)
    self.linear = nn.Linear(self.num_tokens * self.tokenizer.embed_dim, self.tokenizer.num_classes, 
                            dtype=self.dtype, device=self.device)
    self.relu = nn.ReLU()
    self.drop = nn.Dropout(.25)
    self.sm = nn.LogSoftmax(dim=1)

  def forward(self, x):
    tokens = self.tokenizer(x)
    z = self.positional(tokens)
    for k in range(self.num_layers):
      z = self.transformers[k](z)
    z = self.flat(z)
    z = self.relu(self.linear(z))
    z = self.sm(z)
    return z

  def predict(self, x):
    x = self.forward(x)
    return torch.argmax(x, dim=1)

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.tokenizer = self.tokenizer.to(*args, **kwargs)
    self.positional = self.positional.to(*args, **kwargs)
    for k in range(self.num_layers):
      self.transformers[k] = self.transformers[k].to(*args, **kwargs)
    self.linear = self.linear.to(*args, **kwargs)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    self.tokenizer = self.tokenizer.train(*args, **kwargs)
    self.positional = self.positional.train(*args, **kwargs)
    for k in range(self.num_layers):
      self.transformers[k] = self.transformers[k].train(*args, **kwargs)
    self.linear = self.linear.train(*args, **kwargs)
    return self
