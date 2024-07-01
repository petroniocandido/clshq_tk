from pyFTS.models.multivariate import mvfts, wmvfts, granular
import torch
from torch import nn
from torch.utils.data import DataLoader

from clshq_tk.modules.fuzzy import GridPartitioner, Fuzzyfier
from clshq_tk.common import checkpoint, resume, DEVICE, DEFAULT_PATH

big_prime_number = torch.tensor([17461204521323])

def hash_tensor(x):
  b  = x.size(0)
  hash = torch.zeros(b,1).to(torch.int64)
  for b in range(b):
    bx = torch.flatten(x[b,...]).to(torch.int64)
    for t in bx:
      hash[b,:] += t
      hash[b,:] *= big_prime_number
  return hash


class Tokenizer(nn.Module):
  def __init__(self, partitioner, embed_dim, window_size,
               step_size,
               device = None, dtype = torch.float64, **kwargs):
    super().__init__()

    self.partitioner = partitioner
    self.fuzzyfier = Fuzzyfier(self.partitioner, device = device, dtype=dtype, **kwargs)
    self.window_size = window_size
    self.step_size = step_size
    self.embed_dim = embed_dim
    self.vocab = {}
    self.vocab_size = 1
    self.embedding = nn.Embedding(self.partitioner.num_vars * self.partitioner.partitions * window_size,
                                  embed_dim)
    self.device = device
    self.dtype = dtype

  def total_tokens(self, x):
    if len(x.size()) == 3:
      samples = x.size(2)
    else:
      samples = x.size(1)

    inc = 0 if (samples - self.window_size) % self.step_size == 0 else 1
    return ((samples - self.window_size) // self.step_size) + inc

  def sliding_window(self, x):
    samples = x.size(2)
    return [window for window in range(0, samples-self.window_size, self.step_size)]

  def forward(self, x, **kwargs):
    batch, _, samples = x.size()
    x = x.to(self.dtype)

    if samples < self.window_size:
      raise Exception("There are less samples than the window_size")

    num_tokens = self.total_tokens(x)

    tokens = torch.zeros(batch, num_tokens, self.embed_dim, dtype=self.dtype, device = self.device)

    for ix, window in enumerate(self.sliding_window(x)):
      data = x[:,:, window : window + self.window_size]
      fuzz = self.fuzzyfier(data)
      _hash = hash_tensor(fuzz).detach().cpu().numpy()

      for b in range(batch):

        token = int(_hash[b].item())

        if self.training:
          if token in self.vocab:
            num_token = self.vocab[token]
          else:
            self.vocab[token] = self.vocab_size
            num_token = self.vocab_size
            self.vocab_size += 1
        else:
          num_token = self.vocab[token] if token in self.vocab else 0

        tokens[b,ix,:] = self.embedding(torch.tensor(num_token))

    return tokens  


  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.fuzzyfier = self.fuzzyfier.to(*args, **kwargs)
    self.embedding = self.embedding.to(*args, **kwargs)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    return self


def training_loop(model, dataset, **kwargs):
  batch_size = kwargs.get('batch', 10)
  dataloader = DataLoader(dataset.train(), batch_size=batch_size, shuffle=True)
  checkpoint_file = kwargs.get('checkpoint_file', 'modelo.pt')
  model.train()
  for X,_ in dataloader:
    X = X.to(model.device)
    _ = model.forward(X)
  
  model.eval()
  checkpoint(model, checkpoint_file)
