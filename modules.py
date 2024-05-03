import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class TemporalConvolution(nn.Module):
   def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        device=None,
        dtype=None
    ):
      super().__init__()
      if in_channels < 1:
        raise ValueError('in_channels should be greater or equal to 1')
      if out_channels < 1:
        raise ValueError('out_channels should be greater or equal to 1')
      if kernel_size < 1:
        raise ValueError('kernel_size should be greater or equal to 1')
      if stride < 1:
        raise ValueError('Stride should be greater or equal to 1')
      if padding < 0:
        raise ValueError('Padding should be greater or equal to 0')
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.kernel_size = kernel_size
      self.stride = stride
      self.padding = padding
      self.device = device

      self.weights = nn.Parameter(torch.randn(out_channels, in_channels, self.kernel_size))
      self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None


   def forward(self, x):
      if len(x.size()) == 3:
        batch, vars, samples  = x.size()
      else:
        raise ValueError('Wrong tensor format')

      if vars != self.in_channels:
        raise ValueError('Number of input channels is inconsistent with in_channels')

      ######################
      # Zero Padding Deprecated
      ######################
      #pads = torch.zeros(batch, self.in_channels, self.padding, device = self.device)

      # Reflection padding
      pads = x[:,:,0].view(batch,vars,1).repeat(1,1,self.padding)

      x = torch.cat([pads, x], dim=2)

      nlength = (samples - self.kernel_size + self.padding) // self.stride

      output = torch.zeros(batch, self.out_channels, nlength, device = self.device)

      for out_channel in range(self.out_channels):
        for t in range(0, nlength, self.stride):
          tmp = x[:, :, t:t+self.kernel_size]
          w = self.weights[out_channel, :, :]
          out =  torch.sum(tmp * w, (2, 1))
          if not self.bias is None:
            out = out + self.bias[out_channel]
          output[:,out_channel,t] = out

      return output

   def to(self, *args, **kwargs):
      self = super().to(*args, **kwargs)
      self.device = args[0]
      return self

   def freeze(self):
    for param in self.parameters():
      param.requires_grad = False


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


class PositionalEncoder(nn.Module):
  def __init__(self, num_vectors, embed_dim, device = None):
    super().__init__()
    self.num_vectors = num_vectors
    self.embed_dim = embed_dim
    self.embedding = nn.Embedding(num_vectors, embed_dim)
    self.embedding.weight.data =  torch.linspace(-.2,.2,num_vectors).repeat(embed_dim,1).T
    self.device = device

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
    self.device = args[0]
    self.embedding = self.embedding.to(self.device)
    return self


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
