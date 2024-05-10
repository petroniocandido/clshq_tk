# Contrastive LSH Tokenization and Embedding Technique for Time Series Classification

This package proposes a time series tokenization package for attention-based classifiers based in two layers of LSH and a final embedding layer trained with a Triplet Contrastive Loss

This work is intended for Time Series Classification tasks, such that given a family of multivariate time series $Y = \{Y_1, ..., Y_M\}$, where each instance is a discrete-time random process $Y_i = (\Omega, \mathcal{T}, c_i\)$, where $\Omega \in \mathbb{R}^n$ is the domain, $\mathcal{T} \in \mathbb{N}^+$ is the time index with size $T$, and $c_i \in C$ is the class label. Each sample $y_i(t) \in Y_i$, $\forall i = 0\ldots M$ corresponds to a vector $\mathbb{R}^n$, $\forall t = 0\ldots T$. 

The discrete set $C = [c_1, \ldots, c_k]$ represents the set of $k$ class labels of the time series family $Y$.

## Contrastive-LSH Embedding and Tokenizer Model

The tokenizer works by splitting the time series into overlapped patches using a sliding window, parametrized with window length $W$ and the increment length $I_W$. Each patch corresponds to the set of samples $P = \{y_i(t),\ldots,y_i(t+W)\} \in Y_i$ and will be transformed into a token $\tau_i \in \mathbb{R}^E$, where $E$ is the dimensionality of the embedding space of each token.

A time series instance $Y_i$ with $T_i$ samples tokenized with parameters $W = w_i$ and $I_W = iw_i$ will produce $N_T = (T - W) // iw_i$ tokens

The tokenization model is a function $T: Y_i \rightarrow [\tau_0, \ldots, \tau_{N_T}]$ composed of four sequential layers:
  1. **Sample-level hashing**: A set of $H_S$ LSHs based on randomized projections which are applied for each sample $y(t)\in Y$, producing an output $h_s(t) \in \mathbb{N}^{H_W}$
  2. **Patch-level hashing**: A set of $H_W$ LSHs based on randomized projections which are applied for each sample window $h_s(t),\ldots,h_s(t+W)$, producing an output token $h_p(w) \in \mathbb{N}^{H_W}$
  3. **Layer Normalization**: Each token $h_p(w)$ is normalized, such that $h_p(w) \sim \mathcal{N}(0,0.1)$
  4. **Contrastive layer**: A linear layer with $H_W$ inputs and $E$ outputs that transforms the token $h_p(w)$ in the embedded token $\tau_w$
 
The LSH layers are just sampled at model creation and are not trainable, and the final layer is trained with a Triplet Loss contrastive error, such that: 
 
## Attention Classifier

- An Attention Classifier was proposed to assess the performance of the tokenizer, composed of the following layers:
  1. Contrastive-LSH Time Series Tokenization and Embedding that transforms an input time series sample $Y_i$ in a set of $N_T$ tokens $[\tau_0, \ldots, \tau_{N_T}]$
  2. Positional Embedding
  3. $M_T$ Transformers layers with $M_H$ attention-heads and $M_F$ linear units in the feed-forward layer
  4. Classification linear layer with $N_T \times E$ inputs and $k$ outputs
  5. Log-Softmax
 
The model employs a Log Cross Entropy Loss error.
