import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

from sklearn.metrics import accuracy_score

from clshq_tk.common import checkpoint, resume, DEVICE, DEFAULT_PATH

from clshq_tk.modules.positional import PositionalEncoder
from clshq_tk.modules.transformer import Transformer

class TSAttentionClassifier(nn.Module):
  def __init__(self, tokenizer, num_tokens, num_layers, num_heads, feed_forward, 
               num_classes = 2,
               device = None, dtype = torch.float64, **kwargs):
    super().__init__()

    self.num_tokens = num_tokens
    self.num_layers = num_layers
    self.tokenizer = tokenizer
    self.tokenizer.freeze()
    self.device = device
    self.dtype = dtype
    self.positional = PositionalEncoder(self.num_tokens, self.tokenizer.embed_dim, 
                            dtype=self.dtype, device=self.device)
    self.transformers = [Transformer(num_heads, self.num_tokens, self.tokenizer.embed_dim, feed_forward, 
                         dtype=self.dtype, device=self.device) 
                         for k in range(num_layers)]
    self.flat = nn.Flatten(1)
    self.linear = nn.Linear(self.num_tokens * self.tokenizer.embed_dim, num_classes, 
                            dtype=self.dtype, device=self.device)
    self.relu = nn.ReLU()
    self.drop = nn.Dropout(.25)
    self.sm = nn.LogSoftmax(dim=1)

  def forward(self, x):
    tokens = self.tokenizer(x)
    z = self.positional(tokens)
    for k in range(self.num_layers):
      z = self.transformers[k](z)
    z = self.flat(z)
    z = self.relu(self.linear(z))
    z = self.sm(z)
    return z

  def predict(self, x):
    x = self.forward(x)
    return torch.argmax(x, dim=1)

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    if isinstance(args[0], str):
      self.device = args[0]
    else:
      self.dtype = args[0]
    self.tokenizer = self.tokenizer.to(*args, **kwargs)
    self.positional = self.positional.to(*args, **kwargs)
    for k in range(self.num_layers):
      self.transformers[k] = self.transformers[k].to(*args, **kwargs)
    self.linear = self.linear.to(*args, **kwargs)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    self.positional = self.positional.train(*args, **kwargs)
    for k in range(self.num_layers):
      self.transformers[k] = self.transformers[k].train(*args, **kwargs)
    self.linear = self.linear.train(*args, **kwargs)
    return self


def training_loop(DEVICE, dataset, model, display = None, **kwargs):

  if display is None:
    from IPython import display

  batch_size = kwargs.get('batch', 10)

  fig, ax = plt.subplots(1,2, figsize=(15, 5))

  model.to(DEVICE)

  checkpoint_file = kwargs.get('checkpoint_file', 'modelo.pt')

  epochs = kwargs.get('epochs', 10)

  error_train = [0]
  acc_train = [0]
  error_val = [0]
  acc_val = [0]

  train_ldr = DataLoader(dataset.train(), batch_size=batch_size, shuffle=True)
  test_ldr = DataLoader(dataset.test(), batch_size=batch_size, shuffle=True)

  loss = F.nll_loss

  lr = kwargs.get('lr', 0.001)
  optimizer = kwargs.get('optim', optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005))

  start_time = time.time()

  for epoch in range(epochs):

    if epoch % 5 == 0:
      checkpoint(model, checkpoint_file)

    model.train()              # Habilita o treinamento do modelo

    errors = []
    acc = []
    for X, y in train_ldr:

      X = X.to(DEVICE)
      y = y.to(DEVICE).long()

      y_pred = model.forward(X).float()

      optimizer.zero_grad()

      error = loss(y_pred, y)

      error.backward() #retain_graph=True )
      optimizer.step()

      errors.append(error.cpu().item())

      prediction = torch.argmax(y_pred, dim=1).detach().cpu().numpy()
      classes = np.array(y.cpu().tolist())

      acc.append(accuracy_score(classes, prediction))

    error_train.append(np.mean(errors))
    acc_train.append(np.mean(acc))


    model.eval()

    errors = []
    acc = []
    with torch.no_grad():
      for X, y in test_ldr:
        X = X.to(DEVICE)
        y = y.to(DEVICE).long()

        y_pred = model.forward(X).float()

        error = loss(y_pred, y)

        errors.append(error.cpu().item())

        prediction = torch.argmax(y_pred, dim=1).detach().cpu().numpy()
        classes = np.array(y.cpu().tolist())

        acc.append(accuracy_score(classes, prediction))

      error_val.append(np.mean(errors))
      acc_val.append(np.mean(acc))

    display.clear_output(wait=True)
    ax[0].clear()
    ax[0].plot(error_train, c='blue', label='Train')
    ax[0].plot(error_val, c='red', label='Test')
    ax[0].legend(loc='upper right')
    ax[0].set_title("Loss - Time {} - Epoch {} - Train {} - Test {}".format(round(time.time() - start_time, 0), epoch, round(error_train[-1],2), round(error_val[-1],2)))
    ax[1].clear()
    ax[1].plot(acc_train, c='blue', label='Train')
    ax[1].plot(acc_val, c='red', label='Test')
    ax[1].set_title("Accuracy - Epoch {} - Train {} - Test {}".format(epoch, round(acc_train[-1],2), round(acc_val[-1],2)))
    ax[1].legend(loc='upper left')
    plt.tight_layout()
    display.display(plt.gcf())

  plt.savefig(DEFAULT_PATH + "training-"+checkpoint_file+".pdf", dpi=150)
  checkpoint(model, checkpoint_file)