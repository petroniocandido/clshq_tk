

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


#from IPython import display

def tokenizer_training_loop(DEVICE, dataset, model, **kwargs):

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

      errors, errors_val = quantizer_train_step(DEVICE, quantizer_train_ldr, quantizer_test_ldr, model, quantizer_loss, quantizer_optimizer,
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

  plt.savefig(DIRETORIO_PADRAO + "training-"+file+".pdf", dpi=150)
  checkpoint(model, checkpoint_file)

  #return curva_treino, curva_teste

from sklearn.metrics import accuracy_score

def classifier_training_loop(DEVICE, dataset, model, **kwargs):

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

      error.backward(retain_graph=True )
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
    ax[0].set_title("Loss - Epoch {}".format(epoch))
    ax[1].clear()
    ax[1].plot(acc_train, c='blue', label='Train')
    ax[1].plot(acc_val, c='red', label='Test')
    ax[1].set_title("Accuracy - Epoch {}".format(epoch))
    ax[1].legend(loc='upper left')
    plt.tight_layout()
    display.display(plt.gcf())

  plt.savefig(DIRETORIO_PADRAO + "training-"+checkpoint_file+".pdf", dpi=150)
  checkpoint(model, checkpoint_file)
