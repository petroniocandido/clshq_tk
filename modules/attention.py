import numpy as np
import pandas as pd

import torch
from torch import nn


def f_token_level(x,y):
  return x @ y

token_level = torch.func.vmap(f_token_level, in_dims=0)

def f_sequence_level(x,t,w):
  return token_level(x, w.repeat(t,1,1))

def f_batch_level(x, w):
  return token_level(x, w.repeat(x.size(0),1,1))


class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, num_tokens, embed_dim, 
               device = None, dtype = torch.float64, **kwargs):
    super().__init__()

    self.num_heads = num_heads
    self.num_tokens = num_tokens
    self.embed_dim = embed_dim
    self.dk = kwargs.get('dk', self.embed_dim)
    self.dv = kwargs.get('dv', self.embed_dim)
    self.device = device
    self.dtype = dtype

    self.sm = nn.Softmax(1)

    self.WQ = [nn.Parameter(torch.randn(self.embed_dim, self.dk, device = self.device, dtype = self.dtype)) 
               for i in range(self.num_heads)]
    self.WK = [nn.Parameter(torch.randn(self.embed_dim, self.dk, device = self.device, dtype = self.dtype)) 
               for i in range(self.num_heads)]
    self.WV = [nn.Parameter(torch.randn(self.embed_dim, self.dv, device = self.device, dtype = self.dtype)) 
               for i in range(self.num_heads)]

    self.WO = nn.Parameter(torch.randn(self.num_heads * self.dv, self.embed_dim, device = self.device, dtype = self.dtype))

  def forward(self, x):
    b, t, e = x.size()

    if t != self.num_tokens:
      raise Exception("Number of tokens different from num_tokens")

    if e != self.embed_dim:
      raise Exception("Token dimension different from embed_dim")
    
    x = x.to(self.dtype)

    Z = torch.zeros(b, self.num_heads, self.num_tokens, self.dv, device = self.device, dtype = self.dtype)
    Z2 = torch.zeros(b, self.num_tokens, self.embed_dim, device = self.device, dtype = self.dtype)
    for h in range(self.num_heads):
      Q_seq_fun = lambda x : f_sequence_level(x, self.num_tokens, self.WQ[h])
      Q_sequence_level = torch.func.vmap(Q_seq_fun, in_dims=0)
      Q = Q_sequence_level(x)

      K_seq_fun = lambda x : f_sequence_level(x, self.num_tokens, self.WK[h])
      K_sequence_level = torch.func.vmap(K_seq_fun, in_dims=0)
      K = K_sequence_level(x)

      V_seq_fun = lambda x : f_sequence_level(x, self.num_tokens, self.WV[h])
      V_sequence_level = torch.func.vmap(V_seq_fun, in_dims=0)
      V = V_sequence_level(x)

    #  Q = torch.zeros(b, self.num_tokens, self.dk, device = self.device, dtype = self.dtype)
    #  K = torch.zeros(b, self.num_tokens, self.dk, device = self.device, dtype = self.dtype)
    #  V = torch.zeros(b, self.num_tokens, self.dv, device = self.device, dtype = self.dtype)
    #  for i in range(self.num_tokens):
    #    xtemp = x[:,i,:].view(b,e)
    #    Q[:,i,:] = xtemp @ self.WQ[h]
    #    K[:,i,:] = xtemp @ self.WK[h]
    #    V[:,i,:] = xtemp @ self.WV[h]

      scores = Q @ K.view(b,e,t)

      A = self.sm(scores / K.size(1) ** 0.5)

      Z[:, h, :, :] = A @ V

    Z_batch_fun = lambda input : f_batch_level(input, self.WO)
    Z_batch_level = torch.func.vmap(Z_batch_fun, in_dims=0)
    Zt = Z.reshape(b, self.num_tokens, self.num_heads * self.dv)
    Z2 = Z_batch_level(Zt)
    
    #Z2 = torch.zeros(b, self.num_tokens, self.embed_dim, device = self.device, dtype = self.dtype)
    #for bb in range(b):
    #  z = Z[bb,:,:,:].reshape(self.num_tokens, self.num_heads * self.dv)
    #  Z2[bb, :, :] = z @ self.WO

    return Z2

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]

    for h in range(self.num_heads):
      self.WQ[h].to(*args, **kwargs)
      self.WK[h].to(*args, **kwargs)
      self.WV[h].to(*args, **kwargs)
    self.WO.to(*args, **kwargs)
    return self