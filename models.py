

class TSTokenizer(nn.Module):
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


class TSAttentionClassifier(nn.Module):
  def __init__(self, tokenizer, num_tokens, device = None, **kwargs):
    super().__init__()

    self.num_tokens = num_tokens
    self.tokenizer = tokenizer
    self.tokenizer.freeze()
    self.device = device
    self.positional = PositionalEncoder(self.num_tokens, self.tokenizer.embed_dim)
    self.attention = MultiHeadAttention(self.tokenizer.num_classes, self.num_tokens, self.tokenizer.embed_dim)
    self.ln = nn.LayerNorm(self.tokenizer.embed_dim)
    self.flat = nn.Flatten(1)
    self.linear1 = nn.Linear(self.num_tokens * self.tokenizer.embed_dim, self.num_tokens * self.tokenizer.num_classes)
    self.linear2 = nn.Linear(self.num_tokens * self.tokenizer.num_classes, self.num_tokens * self.tokenizer.num_classes)
    self.linear3 = nn.Linear(self.num_tokens * self.tokenizer.num_classes, self.tokenizer.num_classes)
    self.gelu = nn.GELU()
    self.relu = nn.ReLU()
    self.drop = nn.Dropout(.25)
    self.sm = nn.LogSoftmax(dim=1)

  def forward(self, x):
    tokens = self.tokenizer(x)
    tokens = self.positional(tokens)
    z = self.attention(tokens)
    z = self.ln(tokens + z)
    z = self.flat(z)
    z = self.gelu(self.linear1(self.drop(z)))
    z = self.gelu(self.linear2(self.drop(z)))
    z = self.relu(self.linear3(z))
    z = self.sm(z)
    return z

  def predict(self, x):
    x = self.forward(x)
    return torch.argmax(x, dim=1)

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.device = args[0]
    self.tokenizer = self.tokenizer.to(self.device)
    self.positional = self.positional.to(self.device)
    self.attention = self.attention.to(self.device)
    self.linear1 = self.linear1.to(self.device)
    self.linear2 = self.linear2.to(self.device)
    return self

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    self.tokenizer = self.tokenizer.train(*args, **kwargs)
    self.positional = self.positional.train(*args, **kwargs)
    self.attention = self.attention.train(*args, **kwargs)
    self.linear1 = self.linear1.train(*args, **kwargs)
    self.linear2 = self.linear2.train(*args, **kwargs)
    return self
