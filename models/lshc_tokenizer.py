import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from clshq_tk.common import checkpoint, resume, DEVICE, DEFAULT_PATH
from clshq_tk.losses import ContrastiveLoss, TripletLoss, NPairLoss, QuantizerLoss, AngularLoss
from clshq_tk.models import lsh_tokenizer
from clshq_tk.modules.lsh import LSH

class Tokenizer(lsh_tokenizer.Tokenizer):
  def __init__(self, num_variables, num_classes, sample_dim, patch_dim, window_size,
               step_size, embed_dim, sample_width = 1, patch_width = 1, 
               device = None, dtype = torch.float64, **kwargs):
    super(Tokenizer, self).__init__(num_variables, num_classes, sample_dim, patch_dim, window_size,
               step_size, sample_width=sample_width, patch_width=patch_width, device=device, dtype=dtype, **kwargs)
    
    self.embed_dim = embed_dim

    self.linear = nn.Linear(self.patch_dim, self.embed_dim)

  def forward(self, x, **kwargs):
    _, _, samples = x.size()
    x = x.to(self.dtype)

    if samples < self.window_size:
      raise Exception("There are less samples than the window_size")

    new_samples = self.encode(x)
    tokens = self.quantize(new_samples)

    tokens = self.norm(tokens.float())

    for tk in range(self.total_tokens(x)):
      tokens[:,tk] = self.linear(tokens[:,tk].clone())

    return tokens

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.linear = self.linear.to(*args, **kwargs)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    self.linear = self.linear.train(*args, **kwargs)
    return self

  def freeze(self):
    for param in self.parameters(recurse=True):
      param.requires_grad = False


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

  fig, ax = plt.subplots(2,2, figsize=(15, 5))

  model.to(DEVICE)

  checkpoint_file = kwargs.get('checkpoint_file', 'modelo.pt')

  epochs = kwargs.get('epochs', 10)
  lr = kwargs.get('lr', 0.001)
  optimizer = kwargs.get('optim', optim.Adam(model.linear.parameters(), lr=lr, weight_decay=0.0005))

  dataset.contrastive_type = None
  
  lsh_train_ldr = DataLoader(dataset.train(), batch_size=batch_size, shuffle=True)

  model.train()
  for X,_ in lsh_train_ldr:
    X = X.to(model.device)
    samples = model.encode(X)
    _ = model.quantize(samples)

  model.eval()

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

  for epoch in range(epochs):

    if epoch % 5 == 0:
      checkpoint(model, checkpoint_file)

    errors, errors_val = train_step(DEVICE, metric_type, train_ldr, test_ldr, model, loss, optimizer)

    error_train.append(np.mean(errors))
    error_val.append(np.mean(errors_val))

    display.clear_output(wait=True)
    ax[0][0].clear()
    ax[0][0].plot(error_train, c='blue', label='Train')
    ax[0][0].plot(error_val, c='red', label='Test')
    ax[0][0].legend(loc='upper left')
    ax[0][0].set_title("Encoder Loss - All Epochs {}".format(epoch))
    ax[0][1].clear()
    ax[0][1].plot(error_train[-20:], c='blue', label='Train')
    ax[0][1].plot(error_val[-20:], c='red', label='Test')
    ax[0][1].set_title("Encoder Loss - Last 20 Epochs {}".format(epoch))
    ax[0][1].legend(loc='upper left')
    plt.tight_layout()
    display.display(plt.gcf())
  
  plt.savefig(DEFAULT_PATH + "training-"+checkpoint_file+".pdf", dpi=150)
  checkpoint(model, checkpoint_file)
