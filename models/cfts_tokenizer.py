import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from clshq_tk.modules.fuzzy import GridPartitioner, trimf, training_loop as grid_training_loop
from clshq_tk.losses import ContrastiveLoss, TripletLoss, NPairLoss, QuantizerLoss, AngularLoss
from clshq_tk.common import checkpoint, resume, DEVICE, DEFAULT_PATH

class Tokenizer(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.device = kwargs.get('device',None)
    self.dtype = kwargs.get('dtype',torch.float64)
    self.membership_function = kwargs.get('mf', trimf)
    self.partitions = kwargs.get('npart', 3)
    self.alpha_cut = kwargs.get('alpha_cut', 0.1)
    self.vars = kwargs.get('vars', 1)
    self.k = kwargs.get('k', 2)
    self.window_size = kwargs.get('window_size', 1)
    self.step_size = kwargs.get('step_size', 1)
    self.embed_dim = kwargs.get('embed_dim', 20)

    self.partitioner = GridPartitioner(self.membership_function, self.partitions,self.vars,
                                       device=self.device, dtype=self.dtype)
    
    in_len = self.vars * self.window_size * self.k

    self.linear = nn.Sequential(
      nn.Dropout(.25),
      nn.Linear(in_len, in_len), 
      nn.Linear(in_len, self.embed_dim),
      nn.LayerNorm(self.embed_dim)
    )
    

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
      fuzz = self.partitioner(data, mode = 'indexes', k = self.k)
      flat = torch.flatten(fuzz,start_dim=1)
      z = self.linear(flat)
      tokens[:,ix,:] = z 

    return tokens  
  
  def freeze(self):
    self.training = False
    self.linear.requires_grad_ = False


  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.partitioner = self.partitioner.to(*args, **kwargs)
    self.linear = self.linear.to(*args, **kwargs)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    return self


def train_step(DEVICE, metric_type, train, test, model, loss, optim):
  
  model.linear.train()

  errors = []
  for out in train:

    optim.zero_grad()

    if metric_type == "contrastive":
      Xa, ya, Xb, yb = out
      Xa = Xa.to(DEVICE)
      Xb = Xb.to(DEVICE)
      ya = ya.to(DEVICE)
      yb = yb.to(DEVICE)

      a_pred = model(Xa)
      b_pred = model(Xb)

      tt = model.total_tokens(Xa)

      tmp = torch.zeros(tt).to(DEVICE)
      for ix in range(tt):
        tmp[ix] = loss(a_pred[:,ix,:], ya, b_pred[:,ix,:], yb)
      error = tmp.mean()

    elif metric_type in ("triplet", "npair", "angular"):
      Xa, ya, Xp, _, Xn, _ = out
      Xa = Xa.to(DEVICE)
      Xp = Xp.to(DEVICE)
      Xn = Xn.to(DEVICE)

      tt = model.total_tokens(Xa)

      a_pred = model(Xa)
      p_pred = model(Xp)

      if metric_type in  ("triplet", "angular"):
        n_pred = model(Xn)
      else:
        batch, nvars, samples = a_pred.size()                #anchor
        batch, nlabels, nvars, samples = Xn.size()           #negatives
        n_pred = torch.zeros(batch, nlabels, tt, model.embed_dim).to(DEVICE)
        for label in range(nlabels):
          n_pred[:,label,:,:] = model(Xn[:,label,:,:].resize(batch,nvars,samples))
        n_pred = n_pred.to(DEVICE)

      tmp = torch.zeros(tt).to(DEVICE)
      for token in range(tt):
        if metric_type in  ("triplet", "angular"):
          tmp[token] = loss(a_pred[:,token,:], p_pred[:,token,:], n_pred[:,token,:])
        else:
          tmp[token] = loss(a_pred[:,token,:], p_pred[:,token,:], n_pred[:,:,token,:])

      error = tmp.mean()

    error.backward()
    optim.step()

    # Grava as métricas de avaliação
    errors.append(error.cpu().item())


  ##################
  # VALIDATION
  ##################

  model.linear.eval()

  errors_val = []
  with torch.no_grad():
    for out in test:
      if metric_type == "contrastive":
        Xa, ya, Xb, yb = out
        Xa = Xa.to(DEVICE)
        Xb = Xb.to(DEVICE)
        ya = ya.to(DEVICE)
        yb = yb.to(DEVICE)

        a_pred = model(Xa)
        b_pred = model(Xb)

        tt = model.total_tokens(Xa)
        tmp = torch.zeros(tt).to(DEVICE)
        for ix in range(tt):
          tmp[ix] = loss(a_pred[:,ix,:], ya, b_pred[:,ix,:], yb)
        error_val = tmp.mean()

      elif metric_type in ("triplet", "npair", "angular"):
        Xa, ya, Xp, _, Xn, _ = out
        Xa = Xa.to(DEVICE)
        Xp = Xp.to(DEVICE)
        Xn = Xn.to(DEVICE)

        tt = model.total_tokens(Xa)

        a_pred = model(Xa)
        p_pred = model(Xp)

        if metric_type in  ("triplet", "angular"):
          n_pred = model(Xn)
        else:
          batch, nvars, samples = a_pred.size()                #anchor
          batch, nlabels, nvars, samples = Xn.size()           #negatives
          n_pred = torch.zeros(batch, nlabels, tt, model.embed_dim).to(DEVICE)
          for label in range(nlabels):
            n_pred[:,label,:,:] = model(Xn[:,label,:,:].resize(batch,nvars,samples)) # CHECK IT
          n_pred = n_pred.to(DEVICE)

        tmp = torch.zeros(tt).to(DEVICE)
        for token in range(tt):
          if metric_type in  ("triplet", "angular"):
            tmp[token] = loss(a_pred[:,token,:], p_pred[:,token,:], n_pred[:,token,:])
          else:
            tmp[token] = loss(a_pred[:,token,:], p_pred[:,token,:], n_pred[:,:,token,:]) # CHECK IT

        error_val = tmp.mean()

      errors_val.append(error_val.cpu().item())

  return errors, errors_val


def training_loop(DEVICE, dataset, model, display = None, **kwargs):

  if display is None:
    from IPython import display

  metric_type = dataset.contrastive_type

  batch_size = kwargs.get('batch', 10)

  fig, ax = plt.subplots(1,2, figsize=(15, 5))

  model.to(DEVICE)

  checkpoint_file = kwargs.get('checkpoint_file', 'modelo.pt')

  epochs = kwargs.get('epochs', 10)
  lr = kwargs.get('lr', 0.001)
  optimizer = kwargs.get('optim', optim.Adam(model.linear.parameters(), lr=lr, weight_decay=0.0005))

  dataset.contrastive_type = None
  
  grid_training_loop(model.partitioner, dataset)

  dataset.contrastive_type = metric_type

  train_ldr = DataLoader(dataset.train(), batch_size=batch_size, shuffle=True)
  test_ldr = DataLoader(dataset.test(), batch_size=batch_size, shuffle=True)
  
  if metric_type == "contrastive":
    loss = ContrastiveLoss()
  elif metric_type == "triplet":
    loss = TripletLoss()
  elif metric_type == "angular":
    loss = AngularLoss()
  elif  metric_type == "npair":
    loss = NPairLoss()

  error_train = []
  error_val = []

  start_time = time.time()

  for epoch in range(epochs):

    if epoch % 5 == 0:
      checkpoint(model, checkpoint_file)

    errors, errors_val = train_step(DEVICE, metric_type, train_ldr, test_ldr, model, loss, optimizer)

    error_train.append(np.mean(errors))
    error_val.append(np.mean(errors_val))

    display.clear_output(wait=True)
    ax[0].clear()
    ax[0].plot(error_train, c='blue', label='Train')
    ax[0].plot(error_val, c='red', label='Test')
    ax[0].legend(loc='upper left')
    ax[0].set_title("Tokenizer Loss - All Epochs {} - Time: {} s".format(epoch, round(time.time() - start_time, 0)))
    ax[1].clear()
    ax[1].plot(error_train[-20:], c='blue', label='Train')
    ax[1].plot(error_val[-20:], c='red', label='Test')
    ax[1].set_title("Tokenizer Loss - Last 20 Epochs".format(epoch))
    ax[1].legend(loc='upper left')
    plt.tight_layout()
    display.display(plt.gcf())
  
  plt.savefig(DEFAULT_PATH + "training-"+checkpoint_file+".pdf", dpi=150)
  checkpoint(model, checkpoint_file)