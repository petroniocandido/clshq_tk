
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
from datetime import date
import copy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_PATH = ''

def checkpoint(model, file):
  torch.save(model.state_dict(), DEFAULT_PATH + file)

def checkpoint_all(model, optimizer, file):
  torch.save({
    'optim': optimizer.state_dict(),
    'model': model.state_dict(),
}, DEFAULT_PATH + file)

def resume(model, file):
  model.load_state_dict(torch.load(DEFAULT_PATH + file, map_location=torch.device(DEVICE)))

def resume_all(model, optimizer, file):
  checkpoint = torch.load(DEFAULT_PATH + file, map_location=torch.device(DEVICE))
  model.load_state_dict(checkpoint['model'])
  if not optimizer is None:
    optimizer.load_state_dict(checkpoint['optim'])

def classification_metrics(dataloader, model):
  model.to(DEVICE)
  model.eval()
  model.double()

  acc = []
  prec = []
  rec = []
  f1 = []
  for X,y in dataloader:
    X = X.to(DEVICE)
    prediction = model.predict(X).cpu().numpy()
    classes = np.array(y.cpu().tolist())

    acc.append(accuracy_score(classes, prediction))
    prec.append(precision_score(classes, prediction, average='macro'))
    rec.append(recall_score(classes, prediction, average='macro'))
    f1.append(f1_score(classes, prediction, average='macro'))

  return pd.DataFrame([[np.mean(acc), np.mean(prec), np.mean(rec), np.mean(f1)]], columns=['Acc','Prec','Rec','F1'])

def plot_token_space(model, dataset, file):
  model.eval()
  nd = model.embed_dim
  fig, ax = plt.subplots(nd,nd, figsize=(15, 10))

  model = model.to('cpu')

  if model.quantizer_type == 'quant':
    vectors = [model.quantizer.embedding(torch.tensor(i)).detach().numpy() for i in range(model.quantizer.num_vectors)]
  else:
    vectors = [k.detach().numpy() for k in model.quantizer.vectors.values()]

  vectors = np.array(vectors)
  for x in range(nd):
    for y in range(nd):
      ax[x][y].scatter(vectors[:,x],vectors[:,y])

  plt.tight_layout()

  plt.savefig(DEFAULT_PATH + "token-space-"+file+".pdf", dpi=150)


def plot_token_space_usage(model, dataset, file):
  model.eval()
  nd = model.embed_dim
  fig, ax = plt.subplots(1,3, figsize=(15, 5))
  colors = ['blue', 'red', 'green', 'orange']
  for ix in range(dataset.num_instances):
    data = dataset[ix]
    x = data[0].view(1,6,30)
    y = int(data[1].detach().item())
    tokens = model(x)
    vec = tokens.squeeze().detach().numpy()
    ax[0].plot(vec[:,0], vec[:,1], c=colors[y])
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[1].plot(vec[:,1], vec[:,2], c=colors[y])
    ax[1].set_xlabel("Y")
    ax[1].set_ylabel("Z")
    ax[2].plot(vec[:,0], vec[:,2], c=colors[y])
    ax[2].set_xlabel("X")
    ax[2].set_ylabel("Z")
  plt.tight_layout()
  plt.savefig(DEFAULT_PATH + "3D-tokenspace-"+file+".pdf", dpi=150)

def plot_token_usage(tok, dataset, file):
  tok.eval()
  rows = []

  tok.quantizer.clear_statistics()

  for ix in range(dataset.num_instances):
    data = dataset[ix]
    x = data[0].view(1,6,30)
    y = int(data[1].detach().item())
    indexes = tok(x, return_index=True)
    row = indexes.T.squeeze().detach().numpy().flatten()
    row = np.hstack([row, y])
    rows.append(row)

  columns = ['Tok'+str(k+1) for k in range(10) ]
  columns.append('Class')

  df = pd.DataFrame(rows,columns=columns)

  rows = []
  for label in range(dataset.num_labels):
    row = []
    tmp = df[(df['Class'] == label)]
    for k in range(10):
      m = tmp['Tok'+str(k+1)].values.mean()
      row.append(m)
    row.append(label)
    rows.append(row)

  df2 = pd.DataFrame(rows,columns=columns)

  colors = ['blue', 'red', 'green', 'orange']

  fig, ax = plt.subplots(3, 1, figsize=(15, 10))

  if tok.quantizer_type == 'quant':
    stat = [k for k in tok.quantizer.statistics.values()]
    norm = np.sum(stat)
    ax[0].bar([k for k in tok.quantizer.statistics.keys()], stat/norm)
  else:
    if tok.quantizer.dim == 1:
      ks = sorted([k for k in tok.quantizer.statistics.keys()])
      stat = [tok.quantizer.statistics[k] for k in ks]
      norm = np.sum(stat)
      ax[0].bar([k for k in ks], stat/norm)
    elif tok.quantizer.dim == 2:
      ks = sorted([k for k in tok.quantizer.statistics.keys()])
      stat = [tok.quantizer.statistics[k] for k in ks]
      norm = np.sum(stat)

      if tok.quantizer.dim == 1:
        plt.bar([k for k in ks], stat/norm)
      else:
        kss = np.array(ks)
        _minx = int(min(kss[:,0]))
        _maxx = int(max(kss[:,0]))
        xrange = _maxx - _minx
        _miny = int(min(kss[:,1]))
        _maxy = int(max(kss[:,1]))
        yrange = _maxy - _miny
        mat = np.zeros((xrange, yrange))
        for k in ks:
          x,y = k
          x = int(x+abs(_minx)) - 1
          y = int(y+abs(_miny)) - 1
          mat[x,y] = tok.quantizer.statistics[k]/norm
        ax[0].matshow(mat)
    ax[0].set_title('Max: {}  Mean: {}'.format(max(stat), np.mean(stat)))



  ax[0].set_title("Token usage distribution")

  ax[1] = parallel_coordinates(df, 'Class', ax = ax[1], color=colors)
  lgd1 = ax[0].legend(loc='center left', bbox_to_anchor=(1., .5), title="Class")
  ax[1].set_title("Token usage by instance and class")
  if tok.quantizer_type == 'quant':
    ax[1].set_ylim([0, tok.quantizer.num_vectors])
  else:
    if tok.quantizer.dim == 1:
      ks = [k for k in tok.quantizer.vectors.keys()]
      _min = min(ks)
      _max = max(ks)
      ax[1].set_ylim([_min, _max])

  ax[2] = parallel_coordinates(df2, 'Class', ax = ax[2], color=colors, linewidth=2.5)
  lgd1 = ax[2].legend(loc='center left', bbox_to_anchor=(1., .5), title="Class")
  ax[2].set_title("Average token usage by class")
  if tok.quantizer_type == 'quant':
    ax[2].set_ylim([0, tok.quantizer.num_vectors])
  else:
    if tok.quantizer.dim == 1:
      ks = [k for k in tok.quantizer.vectors.keys()]
      _min = min(ks)
      _max = max(ks)
      ax[2].set_ylim([_min, _max])

  plt.tight_layout()

  plt.savefig(DEFAULT_PATH + "token-usage-"+file+".pdf", dpi=150)
