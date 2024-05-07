import numpy as np
import pandas as pd

import torch
from torch import nn

class LSH(nn.Module):
  def __init__(self, embed_dim, width = 1000, num_dim = 1, 
               device = None, dtype = torch.float64):
    super().__init__()
    self.dim = num_dim
    self.device = device
    self.dtype = dtype

    if self.dim == 1:
      self.weights = nn.Parameter(torch.rand(embed_dim, dtype = self.dtype, device = self.device) * width, 
                                  requires_grad = False)
    else:
      self.weights = [nn.Parameter(torch.rand(embed_dim, dtype = self.dtype, device = self.device) * width, 
                                   requires_grad = False) for k in range(self.dim)]

    self.vectors = {}
    self.statistics = {}
    self.tensors = torch.tensor(0)

  def tensorize(self):
    tmp = np.array(sorted([k for k in self.vectors.keys()]))
    self.tensors = torch.from_numpy(tmp).int().to(self.device)

  def forward(self, x, **kwargs):
    return_index = kwargs.get('return_index', False)
    nearest_neighbor = kwargs.get('nearest_neighbor', False)
    batch = x.size(0)

    x = x.to(self.dtype)

    if self.dim == 1:
      v = torch.trunc(self.weights @ x.T).int()
    else:
      v = torch.zeros(batch, self.dim, device=self.device).int()
      for nd in range(self.dim):
        v[:,nd] = torch.trunc(self.weights[nd] @ x.T).int()

    if not return_index:
      ret = torch.zeros(x.size(), device=self.device).int()

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
        if nearest_neighbor:
          dist = self.tensors @ v.T
          tmp = self.tensors[torch.argmin(dist.T, dim=1).int()]
          return tmp
        else:
          return v
      else:
        for ct, i in enumerate(v.detach().cpu().numpy()):
          ii = int(i) if self.dim == 1 else tuple(i.tolist())
          ret[ct,:] = self.vectors[ii]
        return ret

    else:
      if return_index:
        if nearest_neighbor:
          dist = self.tensors @ v.T
          tmp = self.tensors[torch.argmin(dist.T, dim=1).int()]
          return tmp
        else:
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


  def clear_statistics(self):
    self.statistics = {k:0 for k in self.statistics.keys()}

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]

    if self.dim == 1:
      self.weights = self.weights.to(*args, **kwargs)
    else:
      for i in range(self.dim):
        self.weights[i] = self.weights[i].to(*args, **kwargs)

    return self