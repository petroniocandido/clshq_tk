import torch
from torch import nn

from clshq_tk.modules.lsh import LSH

class Tokenizer(nn.Module):
  def __init__(self, num_variables, num_classes, embed_dim, window_size,
               step_size, width, device = None, **kwargs):
    super().__init__()

    self.window_size = window_size
    self.step_size = step_size
    self.embed_dim = embed_dim
    self.num_classes = num_classes
    self.device = device

    self.sample_level = LSH(num_variables, width=width, num_dim = embed_dim)
    self.patch_level = LSH(window_size * embed_dim, width=width, num_dim = embed_dim)

  def total_tokens(self, x):
    samples = x.size(2)
    return (samples - self.window_size) // self.step_size

  def sliding_window(self, x):
    samples = x.size(2)
    return [window for window in range(0, samples-self.window_size, self.step_size)]

  def encode(self, x):
    batch, v, samples = x.size()
    new_samples = torch.zeros(batch, samples, self.embed_dim).to(self.device)
    for ix in range(samples):
      data = x[:,:,ix]
      e = self.sample_level(data)
      new_samples[:, ix, :] = e
    return new_samples

  def quantize(self, x, **kwargs):
    num_tokens = self.total_tokens(x)
    batch, samples, embed_dim = x.size()
    tokens = torch.zeros(batch, num_tokens, embed_dim).to(self.device)
    for ix, window in enumerate(self.sliding_window(x)):
      data = x[:,window : window + self.window_size,:].squeeze()
      e = self.patch_level(data)
      tokens[:,ix,:] = e
    return tokens

  def forward(self, x, **kwargs):
    batch, v, samples = x.size()

    if samples < self.window_size:
      raise Exception("There are less samples than the window_size")

    new_samples = self.encode(x)
    tokens = self.quantize(new_samples)

    return tokens

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.device = args[0]
    self.encoder = self.encoder.to(*args, **kwargs)
    self.quantizer = self.quantizer.to(*args, **kwargs)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    self.encoder = self.encoder.train(*args, **kwargs)
    self.quantizer.train(*args, **kwargs)
    return self

  def freeze(self):
    for param in self.parameters(recurse=True):
      param.requires_grad = False

