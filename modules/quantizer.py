import numpy as np
import pandas as pd

import torch
from torch import nn

def gaussian_kernel(x, mean, sd):
  return torch.exp(-(x-mean)**2/(2*sd**2))

def wknn(ix, k, max):
  if ix - k < 0:
    v = [k for k in range(0, ix + k)]
  elif ix + k >= max:
    v =  [k for k in range(ix - k, max)]
  else:
    v =  [k for k in range(ix - k, ix + k + 1)]

  v = torch.Tensor(v).int()
  d = torch.Tensor(v - ix)
  n = torch.Tensor([gaussian_kernel(k, torch.tensor(0), torch.tensor(1)) for k in d])
  w = n/torch.sum(n)
  return v, w


class VectorQuantizer(nn.Module):
  def __init__(self, num_vectors, embed_dim, knn = 20, device = None):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_vectors = num_vectors
    self.embedding = nn.Embedding(num_vectors, embed_dim)
    self.embedding.weight.data = torch.linspace(-1,1,num_vectors).repeat(embed_dim,1).T #.uniform_(-1.0, 1.0)
    self.k = knn
    self.statistics = {k:0 for k in range(num_vectors)}
    self.device = device


  def forward(self, x, **kwargs):
    return_index = kwargs.get('return_index', False)
    batch, _ = x.size()
    dist = self.embedding.weight @ x.T
    ix = torch.argmin(dist.T, dim=1)
    if self.training:
      epoch = kwargs.get('epoch',0)
      epochs = kwargs.get('epochs',1)
      k = self.k - int(self.k * (epoch/epochs))
      ret = torch.zeros(batch, self.embed_dim, device = self.device)
      for b in range(batch):
        out = torch.zeros(self.embed_dim, device = self.device)
        neigs, weights = wknn(ix[b].cpu().numpy(), k, self.num_vectors)
        neigs = neigs.to(self.device)
        weights = weights.to(self.device)
        for i,w in zip(neigs,weights):
          out += self.embedding(i) * w
        ret[b,:] = out
      return ret
    else:
      for i in ix.cpu().numpy():
        self.statistics[int(i)] += 1
      if return_index:
        return ix
      else:
        return self.embedding(ix).view(x.size())

  def clear_statistics(self):
    self.statistics = {k:0 for k in range(self.num_vectors)}

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.embedding.weight.to(*args, **kwargs)
    self.device = args[0]
    return self

  def freeze(self):
    for param in self.parameters(recurse=True):
      param.requires_grad = False