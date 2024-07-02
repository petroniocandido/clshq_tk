import torch
from torch import nn
import math
from torch.utils.data import DataLoader


def trimf(x, parameters):
  a,b,c = parameters
  t1 = (x-a)/(b-a)
  t2 = (c-x)/(c-b)
  t3 = torch.where(t1 < t2, t1, t2)
  return torch.where(t3 > 0, t3, 0)


def trapmf(x, parameters):
  a, b, c, d = parameters
  if a == b and x == a:
    return 1
  elif c == d and x == c:
    return 1
  else:
    return torch.max(torch.min( (x-a)/(b-a), 1,(d-x)/(d-c) ), 0 )


def gaussmf(x, parameters):
  median, sd = parameters
  return torch.exp((-(x - median)**2.0)/(2.0 * sd**2.0))


def var_names(var, max_vars):
  nc = int(math.log(max_vars, 25))+1
  ret = ''
  tmp_var = var
  for i in range(nc, 0, -1):
    div = tmp_var // 25
    mod = tmp_var - (div * 25)
    n = div if i > 1 else mod
    ret = ret + chr(n + 65)
    tmp_var = mod
  return ret


class GridPartitioner(nn.Module):
  def __init__(self, mf, npart, vars, device = None, dtype = torch.float64, **kwargs):
    super().__init__()
    self.membership_function = mf
    self.partitions = torch.tensor(npart)
    self.alpha_cut = torch.tensor(kwargs.get('alpha_cut', 0.1))
    self.device = device
    self.dtype = dtype
    if mf == trimf:
      self.nparam = torch.tensor(3)
    elif mf == trapmf:
      self.nparam = torch.tensor(4)
    elif mf == gaussmf:
      self.nparam = torch.tensor(2)

    self.num_vars = torch.tensor(vars)

    self.fuzzy_sets = torch.zeros(vars, self.partitions, self.nparam, device=self.device)

    self.names = []

    self.lower_bounds = torch.zeros(self.num_vars, device=self.device)
    self.upper_bounds = torch.zeros(self.num_vars, device=self.device)


  def forward(self, data, **kwargs):
    batch, vars, samples = data.size()

    if vars != self.num_vars:
      raise Exception("Wrong number of variables")
    
    changed = False

    for v in range(vars):

      _max = torch.max(data[:,v,:])
      if _max > self.upper_bounds[v]:
        self.upper_bounds[v] = _max
        changed = True
      else:
        _max = self.upper_bounds[v]

      _min = torch.min(data[:,v,:])
      if _min < self.lower_bounds[v]:
        self.lower_bounds[v] = _min
        changed = True
      else:
        _min = self.lower_bounds[v]

      if changed:
        centers = torch.linspace(_min, _max, self.partitions, device=self.device)
        partlen = torch.abs(centers[1] - centers[0])
        for ct, c in enumerate(centers):
          if self.membership_function == trimf:
            self.fuzzy_sets[v, ct,:] = torch.tensor([c - partlen, c, c + partlen], device=self.device)
          elif self.membership_function == gaussmf:
            self.fuzzy_sets[v, ct,:] =  torch.tensor([c, partlen / 3], device=self.device)
          elif self.membership_function == trapmf:
            q = partlen / 2
            self.fuzzy_sets[v, ct,:] =  torch.tensor([c - partlen, c - q, c + q, c + partlen], device=self.device)
        
        changed = False

    return self.fuzzy_sets
  
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.fuzzy_sets = self.fuzzy_sets.to(*args, **kwargs)
    self.alpha_cut = self.alpha_cut.to(*args, **kwargs)
    self.num_vars = self.num_vars.to(*args, **kwargs)
    self.nparam = self.nparam.to(*args, **kwargs)
    return self


def mf(func, x, p, alpha_cut=0.1):
  mv = func(x,p)
  return torch.where(mv >= alpha_cut, mv, 0)


class Fuzzyfier(nn.Module):
  def __init__(self, partitioner, device = None, dtype = torch.float64, **kwargs):
    super().__init__()
    self.partitioner = partitioner
    self.device = device
    self.dtype = dtype

  def forward(self, x, **kwargs):
    batch, vars, samples = x.size()

    #if vars != self.partitioner.num_vars:
    #  raise Exception("Wrong number of variables")

    # Modes: full, top-k, indexes
    fuzzy_mode = kwargs.get('mode','full')
    fuzzy_param = int(kwargs.get('k',2))

    if fuzzy_mode == 'full':
      fuzzy = torch.zeros(batch, vars, samples, self.partitioner.partitions, device=self.device)
    else:
      fuzzy = torch.zeros(batch, vars, samples, fuzzy_param, device=self.device)
     

    vmap_mf = lambda input,weights: mf(self.partitioner.membership_function, input, weights, self.partitioner.alpha_cut)

    mf_level = torch.func.vmap(vmap_mf, in_dims=0)

    f_sample_level = lambda inputs,samples, weights : mf_level(inputs.repeat(samples),weights)

    for var in range(vars):

      f_sl = lambda inputs: f_sample_level(inputs, self.partitioner.partitions, self.partitioner.fuzzy_sets[var,:,:])

      sample_level = torch.func.vmap(f_sl, in_dims=0)

      f_bl = lambda inputs: sample_level(inputs)

      batch_level = torch.func.vmap(f_bl, in_dims=0)

      if fuzzy_mode == 'full':
        fuzzy[:, var, :, :] = batch_level(x[:,var,:])
      elif fuzzy_mode == 'top-k':
        fuzzy[:, var, :, :], _ = torch.topk(batch_level(x[:,var,:]), fuzzy_param, dim=2)
      elif fuzzy_mode == 'indexes':
        _, fuzzy[:, var, :, :] = torch.topk(batch_level(x[:,var,:]), fuzzy_param, dim=2)

    return fuzzy
  
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.partitioner = self.partitioner.to(*args, **kwargs)
    return self
  

def training_loop(model, dataset, **kwargs):
  batch_size = kwargs.get('batch', 10)
  dataloader = DataLoader(dataset.train(), batch_size=batch_size, shuffle=True)
  model.train()
  for X,_ in dataloader:
    X = X.to(model.device)
    _ = model.forward(X)
  
  model.eval()
  