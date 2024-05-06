import numpy as np
import pandas as pd

import torch
from torch import nn

class LSH(nn.Module):
  def __init__(self, embed_dim, width = 1000, num_dim = 1, device = None):
    super().__init__()
    self.dim = num_dim
    if self.dim == 1:
      self.weights = nn.Parameter(torch.rand(embed_dim) * width, requires_grad = False)
    else:
      self.weights = [nn.Parameter(torch.rand(embed_dim) * width, requires_grad = False) for k in range(self.dim)]
    self.device = device
    self.vectors = {}
    self.statistics = {}

  def forward(self, x, **kwargs):
    return_index = kwargs.get('return_index', False)
    batch = x.size(0)

    if self.dim == 1:
      v = torch.trunc(self.weights @ x.T).int()
    else:
      v = torch.zeros(batch, self.dim, device=self.device)
      for nd in range(self.dim):
        v[:,nd] = torch.trunc(self.weights[nd] @ x.T).int()

    if not return_index:
      ret = torch.zeros(x.size(), device=self.device)
    if self.training:
      for ct, i in enumerate(v.detach().cpu().numpy()):
        ii = int(i) if self.dim == 1 else tuple(i.tolist())
        if ii in self.vectors:
          self.vectors[ii] = (self.vectors[ii] + x[ct,:])/2
          self.statistics[ii] += 1
        else:
          self.statistics[ii] = 1
          self.vectors[ii] = x[ct,:]

      if return_index:
        return v
      else:
        for ct, i in enumerate(v.detach().cpu().numpy()):
          ii = int(i) if self.dim == 1 else tuple(i.tolist())
          ret[ct,:] = self.vectors[ii]
        return ret

    else:
      if return_index:
        return v
      else:
        for ct, i in enumerate(v.detach().cpu().numpy()):
          ii = int(i) if self.dim == 1 else tuple(i.tolist())
          if ii in self.vectors:
            ret[ct,:] = self.vectors[ii]
            self.statistics[ii] += 1
          else:
            ret[ct,:] = x[ct,:]
            self.vectors[ii] = x[ct,:]
            self.statistics[ii] = 1

        return ret

    return x

  def clear_statistics(self):
    self.statistics = {k:0 for k in self.statistics.keys()}

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.device = args[0]
    if self.dim == 1:
      self.weights = self.weights.to(*args, **kwargs)
    else:
      for i in range(self.dim):
        self.weights[i] = self.weights[i].to(*args, **kwargs)

    return self