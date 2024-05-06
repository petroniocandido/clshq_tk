import torch
from torch import nn

class TemporalConvolution(nn.Module):
   def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        device=None,
        dtype=None
    ):
      super().__init__()
      if in_channels < 1:
        raise ValueError('in_channels should be greater or equal to 1')
      if out_channels < 1:
        raise ValueError('out_channels should be greater or equal to 1')
      if kernel_size < 1:
        raise ValueError('kernel_size should be greater or equal to 1')
      if stride < 1:
        raise ValueError('Stride should be greater or equal to 1')
      if padding < 0:
        raise ValueError('Padding should be greater or equal to 0')
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.kernel_size = kernel_size
      self.stride = stride
      self.padding = padding
      self.device = device

      self.weights = nn.Parameter(torch.randn(out_channels, in_channels, self.kernel_size))
      self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None


   def forward(self, x):
      if len(x.size()) == 3:
        batch, vars, samples  = x.size()
      else:
        raise ValueError('Wrong tensor format')

      if vars != self.in_channels:
        raise ValueError('Number of input channels is inconsistent with in_channels')

      # Reflection padding
      pads = x[:,:,0].view(batch,vars,1).repeat(1,1,self.padding)

      x = torch.cat([pads, x], dim=2)

      nlength = (samples - self.kernel_size + self.padding) // self.stride

      output = torch.zeros(batch, self.out_channels, nlength, device = self.device)

      for out_channel in range(self.out_channels):
        for t in range(0, nlength, self.stride):
          tmp = x[:, :, t:t+self.kernel_size]
          w = self.weights[out_channel, :, :]
          out =  torch.sum(tmp * w, (2, 1))
          if not self.bias is None:
            out = out + self.bias[out_channel]
          output[:,out_channel,t] = out

      return output

   def to(self, *args, **kwargs):
      self = super().to(*args, **kwargs)
      self.device = args[0]
      return self

   def freeze(self):
    for param in self.parameters():
      param.requires_grad = False


class TCNEncoder(nn.Module):
  def __init__(self, num_variables, num_classes, num_samples, out_dim, lags = 21):
    super().__init__()

    self.tc1 = TemporalConvolution(in_channels = num_variables, out_channels = num_classes * num_variables, kernel_size = lags,  stride = 1, padding = lags,  bias = True)
    self.tc2 = TemporalConvolution(in_channels = num_classes * num_variables, out_channels = num_classes, kernel_size = lags,  stride = 1, padding = lags,  bias = True)

    self.linear = nn.Linear(num_classes * num_samples, out_dim)
    self.flat = nn.Flatten(1)
    self.drop = nn.Dropout1d(.2)
    self.norm = nn.LayerNorm(num_classes * num_samples)  # For keeping vector approx. unit length
    #self.norm = nn.BatchNorm1d(num_classes * num_samples)

  def forward(self, x):

    x = self.tc1(x)
    x = self.tc2(x)
    x = self.flat(x)
    x = self.norm(x)

    x = self.linear(self.drop(x))

    return x

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.tc1.to(*args, **kwargs)
    self.tc2.to(*args, **kwargs)
    return self