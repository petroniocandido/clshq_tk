from aeon.datasets import load_classification

import numpy as np

import torch
from torch.utils.data import Dataset

import copy

class ClassificationTS(Dataset):
  def __init__(self, name, train, contrastive_type = None, 
               dtype = torch.float64, **kwargs):
    super().__init__()

    self.dtype = dtype

    X, y = load_classification(name)

    self.name = name

    self.num_instances, self.num_attributes, self.num_samples = X.shape

    shuffled_indexes = np.random.permutation(self.num_instances)

    X = X[shuffled_indexes]
    y = y[shuffled_indexes]

    self.train_split = train

    self.labels = np.unique(y)

    self.num_labels = len(self.labels)

    self.transform = kwargs.get('transform', None)

    if kwargs.get('string_labels', False):
      classes = { cx : classe for cx,classe in enumerate(self.labels) }
      classes_inv = { classe : cx for cx,classe in classes.items()}
      y = np.array([classes_inv[k] for k in y])
    else:
      y = np.array([int(float(k)) for k in y])

    self.X = torch.from_numpy(X).to(self.dtype)
    self.y = torch.from_numpy(y)
    self.y = self.y.type(torch.LongTensor)  # Targets sempre do tipo Long

    self.labels = torch.unique(self.y, sorted=True)

    self.contrastive_type = contrastive_type

    self.class_indexes = {}

    self._load_indexes()

    self.is_validation = False

  def _load_indexes(self):
    for label in self.labels:
      self.class_indexes[label.item()] = (self.y == label).nonzero().squeeze().numpy()
      #print("Label {}: {} samples".format(label.item(), len(self.class_indexes[label.item()])))

  def train(self) -> Dataset:
    tmp = copy.deepcopy(self)
    tmp.is_validation = False
    tmp.num_instances = self.train_split
    tmp.X = self.X[0:self.train_split]
    tmp.y = self.y[0:self.train_split]
    tmp._load_indexes()
    return tmp

  def test(self) -> Dataset:
    tmp = copy.deepcopy(self)
    tmp.is_validation = True
    tmp.num_instances = self.num_instances - self.train_split
    tmp.X = self.X[self.train_split:]
    tmp.y = self.y[self.train_split:]
    tmp._load_indexes()
    return tmp

  def any_sample(self, index):
    r = np.random.randint(self.num_samples)
    while r == index:
      r = np.random.randint(self.num_samples)
    return r

  def positive_sample(self, index):
    label = self.y[index]
    return self.class_indexes[label.item()][np.random.randint(len(self.class_indexes[label.item()]))]

  def negative_sample(self, index):
    label = self.y[index]
    nlabel = np.random.randint(self.num_labels)
    while nlabel == label.item():
      nlabel = np.random.randint(self.num_labels)
    return self.class_indexes[nlabel][np.random.randint(len(self.class_indexes[nlabel]))]

  def all_negative_samples(self, index):
    plabel = self.y[index]
    indexes = []
    for ct, nlabel in enumerate(self.labels):
      if nlabel != plabel:
        indexes.append(self.class_indexes[nlabel.item()][np.random.randint(len(self.class_indexes[nlabel.item()]))])

    return indexes


  def __getitem__(self, index):

    if self.contrastive_type is None:
      if not self.transform:# or self.is_validation:
        return self.X[index], self.y[index]
      else:
        return self.transform(self.X[index]), self.y[index]

    elif self.contrastive_type == 'contrastive':
      if isinstance(index, int):
        sample = self.any_sample(index)
      else:
        sample = [self.any_sample(ix) for ix in index]

      if not self.transform:# or self.is_validation:
        return self.X[index], self.y[index], \
          self.X[sample], self.y[sample]
      else:
        return self.transform(self.X[index]), self.y[index], \
          self.transform(self.X[sample]), self.y[sample]

    elif self.contrastive_type in ('triplet','angular'):
      if isinstance(index, int):
        positive = self.positive_sample(index)
        negative = self.negative_sample(index)
      else:
        positive = [self.positive_sample(ix) for ix in index]
        negative = [self.negative_sample(ix) for ix in index]

      if not self.transform:# or self.is_validation:
        return self.X[index], self.y[index], \
          self.X[positive], self.y[positive], \
          self.X[negative], self.y[negative],
      else:
        return self.transform(self.X[index]), self.y[index], \
          self.transform(self.X[positive]), self.y[positive], \
          self.transform(self.X[negative]), self.y[negative]

    elif self.contrastive_type == 'npair':
      if isinstance(index, int):
        positive = self.positive_sample(index)
        negative = self.all_negative_samples(index)
      else:
        positive = [self.positive_sample(ix) for ix in index]
        negative = [self.all_negative_samples(ix) for ix in index]

      if not self.transform:# or self.is_validation:
        return self.X[index], self.y[index], \
          self.X[positive], self.y[positive], \
          self.X[negative], self.y[negative]
      else:
        return self.transform(self.X[index]), self.y[index], \
          self.transform(self.X[positive]), self.y[positive], \
          self.transform(self.X[negative]), self.y[negative]

    else:
      raise Exception("Unknown contrastive type")

  def __len__(self):
    return self.num_instances

  def __iter__(self):
    for ix in range(self.num_instances):
      yield self[ix]

  def __str__(self):
    return "Dataset {}: {} labels {} instances {} attributes {} samples".format(self.name, self.num_labels,
                                                                                self.num_instances, self.num_attributes, self.num_samples)

class Noise(object):
  def __init__(self, type='unif', **kwargs):
    self.type = type
    if self.type == 'unif':
      self.min = kwargs.get('min', 0)
      self.max = kwargs.get('max', 1)
      self.range = self.max - self.min
    elif self.type == 'normal':
      self.std = kwargs.get('std', 0.2)
      self.mean = kwargs.get('mean', 0)

  def __call__(self, tensor):
    if self.type == 'unif':
      return tensor + ((torch.rand(tensor.size()) * self.range) + self.min)
    elif self.type == 'normal':
      return tensor + torch.randn(tensor.size()) * self.std + self.mean

  def __repr__(self):
    if self.type == 'normal':
      return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    elif self.type == 'unif':
      return self.__class__.__name__ + '(min={0}, max={1})'.format(self.min, self.max)


class RandomTranslation(object):
  def __init__(self, max):
    self.max = max

  def __call__(self, tensor):
    return tensor + (((torch.rand(1)*2)-1) * self.max).squeeze()

  def __repr__(self):
    return self.__class__.__name__ + '(max={})'.format(self.max)


class DropAndRepeat(object):
  def __init__(self, prob, seq_size = 1):
    self.prob = prob
    self.seq_size = seq_size

  def __call__(self, tensor):
    b, v, s = tensor.size()
    num_drops = int(s * self.prob)
    x = tensor.clone()
    for r in range(num_drops):
      var = torch.randint(0,v)
      pos = torch.randint(0,s - self.seq_size)
      x[:,var, pos:pos + self.seq_size] = x[:, var, pos].repeat(1,self.seq_size)
      
    return x

  def __repr__(self):
    return self.__class__.__name__ + '(prob={},seq={})'.format(self.prob, self.seq_size)
