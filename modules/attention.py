import numpy as np
import pandas as pd

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, num_tokens, embed_dim, device = None, **kwargs):
    super().__init__()

    self.num_heads = num_heads
    self.num_tokens = num_tokens
    self.embed_dim = embed_dim
    self.dk = kwargs.get('dk', self.embed_dim)
    self.dv = kwargs.get('dv', self.embed_dim)
    self.device = device

    self.sm = nn.Softmax(1)

    self.WQ = [nn.Parameter(torch.randn(self.embed_dim, self.dk)) for i in range(self.num_heads)]
    self.WK = [nn.Parameter(torch.randn(self.embed_dim, self.dk)) for i in range(self.num_heads)]
    self.WV = [nn.Parameter(torch.randn(self.embed_dim, self.dv)) for i in range(self.num_heads)]

    self.WO = nn.Parameter(torch.randn(self.num_heads * self.dv, self.embed_dim))

  def forward(self, x):
    b, t, e = x.size()

    if t != self.num_tokens:
      raise Exception("Number of tokens different from num_tokens")

    if e != self.embed_dim:
      raise Exception("Token dimension different from embed_dim")

    Z = torch.zeros(b, self.num_heads, self.num_tokens, self.dv, device = self.device)
    Z2 = torch.zeros(b, self.num_tokens, self.embed_dim, device = self.device)
    for h in range(self.num_heads):
      Q = torch.zeros(b, self.num_tokens, self.dk)
      K = torch.zeros(b, self.num_tokens, self.dk)
      V = torch.zeros(b, self.num_tokens, self.dv)
      for i in range(self.num_tokens):
        xtemp = x[:,i,:].view(b,e)
        Q[:,i,:] = xtemp @ self.WQ[h]
        K[:,i,:] = xtemp @ self.WK[h]
        V[:,i,:] = xtemp @ self.WV[h]

      scores = Q @ K.view(b,e,t)

      A = self.sm(scores / K.size(1) ** 0.5)

      Z[:, h, :, :] = A @ V

    Z2 = torch.zeros(b, self.num_tokens, self.embed_dim, device = self.device)
    for bb in range(b):
      z = Z[bb,:,:,:].reshape(self.num_tokens, self.num_heads * self.dv)
      Z2[bb, :, :] = z @ self.WO

    return Z2

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.device = args[0]
    for h in range(self.num_heads):
      self.WQ[h].to(*args, **kwargs)
      self.WK[h].to(*args, **kwargs)
      self.WV[h].to(*args, **kwargs)
    self.WO.to(*args, **kwargs)
    return self