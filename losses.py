import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.nn import functional as F


def euclidean_distance(a, b):
  return (a - b).pow(2).sum(1)


def norm(a):
  return a.pow(2).sum()


class ContrastiveLoss(nn.Module):
  def __init__(self, m = 2, device = None):
    super().__init__()
    self.device = device
    self.m = torch.tensor(m).to(self.device)

  def forward(self, xa, ya, xb, yb):
    b = xa.size(0)
    dist = euclidean_distance(xa, xb)
    return torch.where(ya == yb, dist, torch.maximum(torch.zeros(b).to(self.device), self.m - dist)).mean()


class TripletLoss(nn.Module):
  def __init__(self, m = 0.2, device = None):
    super(TripletLoss, self).__init__()
    self.device = device
    self.m = m

  def forward(self, xa, xp, xn):
    distp = euclidean_distance(xa, xp)
    distn = euclidean_distance(xa, xn)
    return torch.relu(self.m + distp - distn).mean()

class NPairLoss(nn.Module):
  def __init__(self):
    super().__init__( device = None)
    self.device = device

  def forward(self, xa, xp, xn):
    b,v,e = xn.size()
    positive = torch.exp(torch.sum(xa * xp, 1))
    negatives = torch.zeros(b).to(self.device)
    for i in range(v):
      negatives += torch.exp(torch.sum(xa * xn[:,i,:], 1))
    return -torch.log(positive / (positive + negatives)).mean()

def tan2(x):
  c = torch.cos(2*x)+1
  return -c/c

class AngularLoss(nn.Module):
  def __init__(self, alpha = 40):
    super().__init__()
    self.alpha = torch.tensor(alpha)
    self.atan = tan2(self.alpha)

  def forward(self, xa, xp, xn):
    fapn = 4 * self.atan * (xa + xp).mul(xn).sum(1) - 2 * (1 + self.atan) * ((xa * xp).sum())
    return torch.log(1 + torch.exp(fapn).sum())
    #xc = (xa + xp)/2
    #return (euclidean_distance(xa, xp) - 4 * self.atan * euclidean_distance(xn, xc)).mean()


class QuantizerLoss(nn.Module):
  def __init__(self):
    super(QuantizerLoss, self).__init__()

  def forward(self, xa, xe):
    if len(xa.size()) == 2:
      return euclidean_distance(xa, xe).mean()
    elif len(xa.size()) == 3:
      batch, num_tokens, _ = xa.size()
      tmp = torch.zeros(num_tokens)
      for ix in range(num_tokens):
        tmp[ix] = euclidean_distance(xa, xe).mean()
      return tmp.mean()

