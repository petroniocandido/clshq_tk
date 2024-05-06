import torch
from torch import nn, optim
from torch.nn import functional as F

from clshq_tk.modules.tcn import TCNEncoder
from clshq_tk.modules.lsh import LSH
from clshq_tk.modules.quantizer import VectorQuantizer



class Tokenizer(nn.Module):
  def __init__(self, num_variables, num_classes, num_tokens, embed_dim, lags, window_size,
               step_size, quantizer_type = 'quant', device = None, **kwargs):
    super().__init__()

    if window_size % lags != 0:
      raise Exception("Window parameter must be a multiple of lags parameter")

    if window_size < step_size:
      raise Exception("step_size parameter must be a lesser or equal to window_size")


    self.quantizer_type = quantizer_type
    self.window_size = window_size
    self.step_size = step_size
    self.embed_dim = embed_dim
    self.num_classes = num_classes

    self.device = device

    self.encoder = TCNEncoder(num_variables, num_classes, window_size, embed_dim, lags)
    if quantizer_type == 'quant':
      self.quantizer = VectorQuantizer(num_tokens, embed_dim)
    elif quantizer_type == 'lsh':
      width = kwargs.get('lsh_width',1000)
      num_dim = kwargs.get('lsh_dim',1)
      self.quantizer = LSH(embed_dim, width=width, num_dim = num_dim)

  def total_tokens(self, x):
    samples = x.size(2)
    return (samples - self.window_size) // self.step_size

  def sliding_window(self, x):
    samples = x.size(2)
    return [window for window in range(0, samples-self.window_size, self.step_size)]

  def encode(self, x):
    batch, v, samples = x.size()
    num_tokens = self.total_tokens(x)
    tokens = torch.zeros(batch, num_tokens, self.embed_dim).to(self.device)
    for ix, window in enumerate(self.sliding_window(x)):
      data = x[:,:,window : window + self.window_size]
      e = self.encoder(data)
      tokens[:, ix, :] = e
    return tokens

  def quantize(self, x, **kwargs):
    return_index = kwargs.get('return_index', False)
    batch, num_tokens, embed_dim = x.size()
    if return_index:
      tokens = torch.zeros(batch, num_tokens).to(self.device)
    else:
      tokens = torch.zeros(batch, num_tokens, self.embed_dim).to(self.device)
    for ix in range(num_tokens):
      data = x[:,ix,:]
      q = self.quantizer(data, **kwargs)
      if return_index:
        tokens[:, ix] = q
      else:
        tokens[:, ix, :] = q
    return tokens

  def forward(self, x, **kwargs):
    batch, v, samples = x.size()
    return_index = kwargs.get('return_index', False)

    if samples < self.window_size:
      raise Exception("There are less samples than the window_size")

    num_tokens = self.total_tokens(x)

    if return_index:
      tokens = torch.zeros(batch, num_tokens).to(self.device)
    else:
      tokens = torch.zeros(batch, num_tokens, self.embed_dim).to(self.device)

    for ix, window in enumerate(self.sliding_window(x)):
      data = x[:,:,window : window + self.window_size]
      e = self.encoder(data)
      q = self.quantizer(e, **kwargs)

      if return_index:
        tokens[:, ix] = q
      else:
        tokens[:, ix, :] = q

    return tokens

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.device = args[0]
    self.encoder = self.encoder.to(*args, **kwargs)
    self.quantizer = self.quantizer.to(*args, **kwargs)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    self.quantizer.train(*args, **kwargs)
    return self

  def freeze(self):
    for param in self.parameters(recurse=True):
      param.requires_grad = False