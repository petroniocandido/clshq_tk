
import torch
from torch import nn, optim
from torch.nn import functional as F

from clshq_tk.modules.positional import PositionalEncoder
from clshq_tk.modules.attention import MultiHeadAttention

class TSAttentionClassifier(nn.Module):
  def __init__(self, tokenizer, num_tokens, device = None, **kwargs):
    super().__init__()

    self.num_tokens = num_tokens
    self.tokenizer = tokenizer
    self.tokenizer.freeze()
    self.device = device
    self.positional = PositionalEncoder(self.num_tokens, self.tokenizer.embed_dim)
    self.attention = MultiHeadAttention(self.tokenizer.num_classes, self.num_tokens, self.tokenizer.embed_dim)
    self.ln = nn.LayerNorm(self.tokenizer.embed_dim)
    self.flat = nn.Flatten(1)
    self.linear1 = nn.Linear(self.num_tokens * self.tokenizer.embed_dim, self.num_tokens * self.tokenizer.num_classes)
    self.linear2 = nn.Linear(self.num_tokens * self.tokenizer.num_classes, self.num_tokens * self.tokenizer.num_classes)
    self.linear3 = nn.Linear(self.num_tokens * self.tokenizer.num_classes, self.tokenizer.num_classes)
    self.gelu = nn.GELU()
    self.relu = nn.ReLU()
    self.drop = nn.Dropout(.25)
    self.sm = nn.LogSoftmax(dim=1)

  def forward(self, x):
    tokens = self.tokenizer(x)
    tokens = self.positional(tokens)
    z = self.attention(tokens)
    z = self.ln(tokens + z)
    z = self.flat(z)
    z = self.gelu(self.linear1(self.drop(z)))
    z = self.gelu(self.linear2(self.drop(z)))
    z = self.relu(self.linear3(z))
    z = self.sm(z)
    return z

  def predict(self, x):
    x = self.forward(x)
    return torch.argmax(x, dim=1)

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.device = args[0]
    self.tokenizer = self.tokenizer.to(self.device)
    self.positional = self.positional.to(self.device)
    self.attention = self.attention.to(self.device)
    self.linear1 = self.linear1.to(self.device)
    self.linear2 = self.linear2.to(self.device)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    self.tokenizer = self.tokenizer.train(*args, **kwargs)
    self.positional = self.positional.train(*args, **kwargs)
    self.attention = self.attention.train(*args, **kwargs)
    self.linear1 = self.linear1.train(*args, **kwargs)
    self.linear2 = self.linear2.train(*args, **kwargs)
    return self
