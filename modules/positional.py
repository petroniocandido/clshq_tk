import torch
from torch import nn


class PositionalEncoder(nn.Module):
  def __init__(self, num_vectors, embed_dim, device = None, dtype = torch.float64):
    super().__init__()
    self.device = device
    self.dtype = dtype
    self.num_vectors = num_vectors
    self.embed_dim = embed_dim
    self.embedding = nn.Embedding(num_vectors, embed_dim)
    self.embedding.weight.data =  torch.linspace(-.2,.2,num_vectors).repeat(embed_dim,1).T.to(self.dtype)

  def forward(self, x):
    b, nv, e = x.size()

    if nv != self.num_vectors:
      raise Exception("Number of tokens different from num_vectors={}".format(self.num_vectors))

    if e != self.embed_dim:
      raise Exception("Embedded vectors dimension should be embed_dim={}".format(self.embed_dim))

    for token in range(0, self.num_vectors):
      x[:,token,:] = x[:,token,:] + self.embedding(torch.tensor(token).to(self.device))

    return x

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.embedding = self.embedding.to(*args, **kwargs)
    return self