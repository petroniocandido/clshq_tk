import torch
from torch import nn

from clshq_tk.modules.lsh import LSH

class Tokenizer(nn.Module):
  def __init__(self, num_variables, num_classes, sample_dim, patch_dim, window_size,
               step_size, sample_width = 1, patch_width = 1, 
               device = None, dtype = torch.float64, **kwargs):
    super().__init__()

    self.window_size = window_size
    self.step_size = step_size
    self.sample_dim = sample_dim
    self.patch_dim = patch_dim
    self.num_classes = num_classes
    self.embed_dim = self.patch_dim
    self.device = device
    self.dtype = dtype

    self.sample_level = LSH(num_variables, width=sample_width, num_dim = sample_dim, 
                            dtype=self.dtype, device=self.device)
    self.patch_level = LSH(window_size * sample_dim, width=patch_width, num_dim = patch_dim, 
                            dtype=self.dtype, device=self.device)
    self.norm = nn.LayerNorm(patch_dim)  # For keeping vector approx. unit length

  def total_tokens(self, x):
    samples = x.size(2)
    inc = 0 if (samples - self.window_size) % self.step_size == 0 else 1
    return ((samples - self.window_size) // self.step_size) + inc

  def sliding_window(self, x):
    samples = x.size(2)
    return [window for window in range(0, samples-self.window_size, self.step_size)]

  def encode(self, x):
    batch, v, samples = x.size()
    new_samples = torch.zeros(batch, self.sample_dim, samples, dtype=self.dtype, device = self.device)
    for ix in range(samples):
      data = x[:,:,ix]
      e = self.sample_level(data, return_index = True)
      new_samples[:,:,ix] = e
    return new_samples

  def quantize(self, x, **kwargs):
    num_tokens = self.total_tokens(x)
    batch, _, _ = x.size()
    tokens = torch.zeros(batch, num_tokens, self.patch_dim, dtype=self.dtype, device = self.device)
    for ix, window in enumerate(self.sliding_window(x)):
      data = x[:,:, window : window + self.window_size].reshape(batch, self.window_size * self.sample_dim)
      e = self.patch_level(data, return_index = True)
      tokens[:,ix,:] = e
    return tokens

  def forward(self, x, **kwargs):
    _, _, samples = x.size()
    x = x.to(self.dtype)

    if samples < self.window_size:
      raise Exception("There are less samples than the window_size")

    new_samples = self.encode(x)
    tokens = self.quantize(new_samples)

    tokens = self.norm(tokens.float())

    return tokens

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.sample_level = self.sample_level.to(*args, **kwargs)
    self.patch_level = self.patch_level.to(*args, **kwargs)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    self.sample_level = self.sample_level.train(*args, **kwargs)
    self.patch_level = self.patch_level.train(*args, **kwargs)
    return self

  def freeze(self):
    for param in self.parameters(recurse=True):
      param.requires_grad = False


def training_loop(model, dataloader):
    model.train()
    for X,_ in dataloader:
      _ = model.forward(X)
    
    model.sample_level.tensorize()
    model.patch_level.tensorize()

    model.eval()
