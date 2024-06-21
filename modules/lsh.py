import numpy as np
import pandas as pd

import torch
from torch import nn

def gram_schmidt(A):
  # From: https://zerobone.net/blog/cs/gram-schmidt-orthogonalization/
  n, m = A.size()
  for i in range(m):        
    q = A[:, i] # i-th column of A
        
    for j in range(i):
      q = q - torch.dot(A[:, j], A[:, i]) * A[:, j]
        
    if np.array_equal(q, np.zeros(q.shape)):
      raise Exception("The column vectors are not linearly independent")
        
    # normalize q
    q = q / torch.sqrt(torch.dot(q, q))
        
    # write the vector back in the matrix
    A[:, i] = q
  return A


activations = {
  'trunc': lambda x: torch.trunc(x),
  'identity': lambda x: x,
  'step': lambda x: torch.heaviside(x, torch.tensor([0]))
}

# Functions for vectorize the LSH hashing generation over the batches using torch.vmap

def f_instance_level_map(x,y):
  return torch.sum(x * y).int()

instance_level_map = torch.func.vmap(f_instance_level_map, in_dims=0)

def f_batch_level_map(x, embed_dim, weigths):
  return instance_level_map(x.repeat(embed_dim,1,1), weigths)


class LSH(nn.Module):
  def __init__(self, embed_dim, width = 1000, num_dim = 1, activation = 'trunc',
              device = None, dtype = torch.float64):
    super().__init__()
    self.dim = num_dim
    self.device = device
    self.dtype = dtype
    self.activation = activations[activation]

    self.weights = nn.Parameter(torch.rand(self.dim, embed_dim, dtype = self.dtype, device = self.device) * width, 
                                  requires_grad = False)

    self.vectors = {}
    self.statistics = {}
    self.tensors = torch.tensor(0)

    lsh_batches = lambda input : f_batch_level_map(input, self.dim, self.weights)

    self.batch_level_map = torch.func.vmap(lsh_batches, in_dims=0)

  def tensorize(self):
    tmp = np.array(sorted([k for k in self.vectors.keys()]))
    self.tensors = torch.from_numpy(tmp).int().to(self.device)

  def forward(self, x, **kwargs):
    return_index = kwargs.get('return_index', False)
    nearest_neighbor = kwargs.get('nearest_neighbor', False)
    batch = x.size(0)

    x = x.to(self.dtype)

    v = self.activation( self.batch_level_map(x))

    #v = torch.zeros(batch, self.dim, device=self.device).int()
    #for nd in range(self.dim):
    #  if self.dim == 1:
    #    v[nd] = self.activation(self.weights[nd, :] @ x.T).int()
    #  else:
    #    v[:,nd] = self.activation(self.weights[nd, :] @ x.T).int()


    if not return_index:
      ret = torch.zeros(x.size(), device=self.device).int()

    if self.training:
      for ct, i in enumerate(v.detach().cpu().numpy()):
        ii = tuple(i.tolist())
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
          ii = tuple(i.tolist())
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
          ii = tuple(i.tolist())
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

    self.weights = self.weights.to(*args, **kwargs)
    
    return self