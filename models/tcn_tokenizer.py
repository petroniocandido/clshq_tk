import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from clshq_tk.common import checkpoint, resume, DEVICE, DEFAULT_PATH
from clshq_tk.losses import ContrastiveLoss, TripletLoss, NPairLoss, QuantizerLoss, AngularLoss

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


def encoder_train_step(DEVICE, metric_type, train, test, model, loss, optim):
  model.train()

  errors = []
  for out in train:

    optim.zero_grad()

    if metric_type == "contrastive":
      Xa, ya, Xb, yb = out
      Xa = Xa.to(DEVICE)
      Xb = Xb.to(DEVICE)
      ya = ya.to(DEVICE)
      yb = yb.to(DEVICE)

      a_pred = model.encode(Xa)
      b_pred = model.encode(Xb)

      tt = model.total_tokens(Xa)

      tmp = torch.zeros(tt)
      for ix in range(tt):
        tmp[ix] = loss(a_pred[:,ix,:], ya, b_pred[:,ix,:], yb)
      error = tmp.mean()

    elif metric_type in ("triplet", "npair", "angular"):
      Xa, ya, Xp, yp, Xn, yn = out
      Xa = Xa.to(DEVICE)
      Xp = Xp.to(DEVICE)
      Xn = Xn.to(DEVICE)

      tt = model.total_tokens(Xa)

      a_pred = model.encode(Xa)
      p_pred = model.encode(Xp)

      if metric_type in  ("triplet", "angular"):
        n_pred = model.encode(Xn)
      else:
        batch, nvars, samples = a_pred.size()                #anchor
        batch, nlabels, nvars, samples = Xn.size()           #negatives
        n_pred = torch.zeros(batch, nlabels, tt, model.embed_dim)
        for label in range(nlabels):
          n_pred[:,label,:,:] = model.encode(Xn[:,label,:,:].resize(batch,nvars,samples))
        n_pred = n_pred.to(DEVICE)

      tmp = torch.zeros(tt)
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

  model.eval()

  errors_val = []
  with torch.no_grad():
    for out in test:
      if metric_type == "contrastive":
        Xa, ya, Xb, yb = out
        Xa = Xa.to(DEVICE)
        Xb = Xb.to(DEVICE)
        ya = ya.to(DEVICE)
        yb = yb.to(DEVICE)

        a_pred = model.encode(Xa)
        b_pred = model.encode(Xb)

        tt = model.total_tokens(Xa)
        tmp = torch.zeros(tt)
        for ix in range(tt):
          tmp[ix] = loss(a_pred[:,ix,:], ya, b_pred[:,ix,:], yb)
        error_val = tmp.mean()

      elif metric_type in ("triplet", "npair", "angular"):
        Xa, ya, Xp, yp, Xn, yn = out
        Xa = Xa.to(DEVICE)
        Xp = Xp.to(DEVICE)
        Xn = Xn.to(DEVICE)

        tt = model.total_tokens(Xa)

        a_pred = model.encode(Xa)
        p_pred = model.encode(Xp)

        if metric_type in  ("triplet", "angular"):
          n_pred = model.encode(Xn)
        else:
          batch, nvars, samples = a_pred.size()                #anchor
          batch, nlabels, nvars, samples = Xn.size()           #negatives
          n_pred = torch.zeros(batch, nlabels, tt, model.embed_dim)
          for label in range(nlabels):
            n_pred[:,label,:,:] = model.encode(Xn[:,label,:,:].resize(batch,nvars,samples)) # CHECK IT
          n_pred = n_pred.to(DEVICE)

        tmp = torch.zeros(tt)
        for token in range(tt):
          if metric_type in  ("triplet", "angular"):
            tmp[token] = loss(a_pred[:,token,:], p_pred[:,token,:], n_pred[:,token,:])
          else:
            tmp[token] = loss(a_pred[:,token,:], p_pred[:,token,:], n_pred[:,:,token,:]) # CHECK IT

        error_val = tmp.mean()

      errors_val.append(error_val.cpu().item())

  return errors, errors_val

def quantizer_train_step(DEVICE, train, test, model, loss, optim, epoch, epochs):
  model.train()

  errors = []
  for X,_ in train:
    X = X.to(DEVICE)
    optim.zero_grad()
    embed = model.encode(X)
    pred = model.quantize(embed, epoch=epoch, epochs=epochs)
    error = loss(embed, pred)

    error.backward()
    optim.step()

    # Grava as métricas de avaliação
    errors.append(error.cpu().item())

  ##################
  # VALIDATION
  ##################

  model.eval()

  errors_val = []
  with torch.no_grad():
    for X,_ in test:
      X = X.to(DEVICE)
      embed = model.encode(X)
      pred = model.quantize(embed)
      error_val = loss(embed, pred)

      errors_val.append(error_val.cpu().item())

  return errors, errors_val


def lsh_quantizer_train_step(DEVICE, train, test, model, loss, optim, epoch, epochs):
  
  model.train()

  errors = []
  for X,_ in train:
    model.quantizer.clear_statistics()
    X = X.to(DEVICE)
    embeds = model.encode(X)
    _ = model.quantize(embeds)
  
    stat = [model.quantizer.statistics[k] for k in model.quantizer.statistics.keys()]
    errors.append(np.mean(stat))


  ##################
  # VALIDATION
  ##################

  model.eval()

  errors_val = []
  with torch.no_grad():
    for X,_ in test:
      model.quantizer.clear_statistics()
      X = X.to(DEVICE)
      embeds = model.encode(X)
      _ = model.quantize(embeds)
    
      stat = [model.quantizer.statistics[k] for k in model.quantizer.statistics.keys()]
      errors_val.append(np.mean(stat))

  return errors, errors_val


def training_loop(DEVICE, dataset, model, display = None, **kwargs):

  if display is None:
    from IPython import display

  encoder_loop = kwargs.get('encoder_loop', True)
  quantizer_loop = kwargs.get('quantizer_loop', True)

  metric_type = dataset.contrastive_type

  batch_size = kwargs.get('batch', 10)

  fig, ax = plt.subplots(2,2, figsize=(15, 5))

  model.to(DEVICE)

  checkpoint_file = kwargs.get('checkpoint_file', 'modelo.pt')

  if encoder_loop:

    epochs = kwargs.get('encoder_epochs', 10)

    encoder_train = [0]
    encoder_val = [0]

    encoder_train_ldr = DataLoader(dataset.train(), batch_size=batch_size, shuffle=True)
    encoder_test_ldr = DataLoader(dataset.test(), batch_size=batch_size, shuffle=True)

    if metric_type == "contrastive":
      encoder_loss = ContrastiveLoss()
    elif metric_type == "triplet":
      encoder_loss = TripletLoss()
    elif metric_type == "angular":
      encoder_loss = AngularLoss()
    elif  metric_type == "npair":
      encoder_loss = NPairLoss()

    encoder_lr = kwargs.get('encoder_lr', 0.001)
    encoder_optimizer = kwargs.get('opt1', optim.Adam(model.encoder.parameters(), lr=encoder_lr, weight_decay=0.0005))

    for epoch in range(epochs):

      if epoch % 5 == 0:
        checkpoint(model, checkpoint_file)

      errors, errors_val = encoder_train_step(DEVICE, metric_type, encoder_train_ldr, encoder_test_ldr, model, encoder_loss, encoder_optimizer)

      encoder_train.append(np.mean(errors))
      encoder_val.append(np.mean(errors_val))

      display.clear_output(wait=True)
      ax[0][0].clear()
      ax[0][0].plot(encoder_train, c='blue', label='Train')
      ax[0][0].plot(encoder_val, c='red', label='Test')
      ax[0][0].legend(loc='upper left')
      ax[0][0].set_title("Encoder Loss - All Epochs {}".format(epoch))
      ax[0][1].clear()
      ax[0][1].plot(encoder_train[-20:], c='blue', label='Train')
      ax[0][1].plot(encoder_val[-20:], c='red', label='Test')
      ax[0][1].set_title("Encoder Loss - Last 20 Epochs {}".format(epoch))
      ax[0][1].legend(loc='upper left')
      plt.tight_layout()
      display.display(plt.gcf())

  if quantizer_loop:

    epochs = kwargs.get('quantizer_epochs', 10)

    if model.quantizer_type == 'quant':

      quantizer_lr = kwargs.get('quantizer_lr', 0.01)
      quantizer_optimizer = kwargs.get('opt2', optim.Adam(model.quantizer.parameters(), lr=quantizer_lr, weight_decay=0.0005))
      quantizer_loss = QuantizerLoss()

    dataset.contrastive_type = None

    quantizer_train_ldr = DataLoader(dataset.train(), batch_size=batch_size, shuffle=True)
    quantizer_test_ldr = DataLoader(dataset.test(), batch_size=batch_size, shuffle=True)

    quantizer_train = [0]
    quantizer_val = [0]

    for epoch in range(epochs):

      if epoch % 5 == 0:
        checkpoint(model, checkpoint_file)

      if model.quantizer_type == 'quant':
        errors, errors_val = quantizer_train_step(DEVICE, quantizer_train_ldr, quantizer_test_ldr, model, quantizer_loss, quantizer_optimizer,
                                                  epoch, epochs)
      elif model.quantizer_type == 'lsh':
        errors, errors_val = lsh_quantizer_train_step(DEVICE, quantizer_train_ldr, quantizer_test_ldr, model, None, None,
                                                  epoch, epochs)

      quantizer_train.append(np.mean(errors))
      quantizer_val.append(np.mean(errors_val))

      display.clear_output(wait=True)
      ax[1][0].clear()
      ax[1][0].plot(quantizer_train, c='blue', label='Train')
      ax[1][0].plot(quantizer_val, c='red', label='Test')
      ax[1][0].legend(loc='upper left')
      ax[1][0].set_title("Quantizer Loss - All Epochs {}".format(epoch))
      ax[1][1].clear()
      stat = [k for k in model.quantizer.statistics.values()]
      norm = np.sum(stat)
      ax[1][1].bar([k for k in model.quantizer.statistics.keys()], stat/norm)
      #ax[1][1].plot(quantizer_train[-20:], c='blue', label='Train')
      #ax[1][1].plot(quantizer_val[-20:], c='red', label='Test')
      #ax[1][1].set_title("Quantizer Loss - Last 20 Epochs {}".format(epoch))
      #ax[1][1].legend(loc='upper left')
      plt.tight_layout()
      display.display(plt.gcf())

      model.quantizer.clear_statistics()


    dataset.contrastive_type = metric_type

  plt.savefig(DEFAULT_PATH + "training-"+checkpoint_file+".pdf", dpi=150)
  checkpoint(model, checkpoint_file)