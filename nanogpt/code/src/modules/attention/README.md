# Transformer attention in this project

## Transformer

A Transformer predicts each next token by repeatedly mixing information from
earlier tokens. It does not process text one token at a time like an RNN;
within one layer, it evaluates every allowed token-to-token relationship in
parallel.

This project uses a **decoder-only Transformer**, the family of architectures
behind GPT. Given tokens `x[0:T]`, it learns the next-token distribution
`P(x[t + 1] | x[:t + 1])`.

## Causal self-attention

Each input token embedding is projected into a query (Q), key (K), and value
(V). For one attention head:

```text
scores = Q K^T / sqrt(head_size)
weights = softmax(mask(scores))
output = weights V
```

The causal mask sets scores for future positions to negative infinity before
the softmax. A token can therefore attend to itself and earlier tokens, but
never to a future token. This prevents the model from seeing the answer while
it is trained to predict the next token.

`head_size` is used for scaling because each Q--K dot product has
`head_size` components. It equals `embedding_size` only for single-head
attention.

## Multi-head attention

Rather than one large attention calculation, multi-head attention runs several
smaller attention heads in parallel. Their outputs are concatenated and passed
through a learned output projection. Different heads can learn different kinds
of relationships, such as nearby characters, word boundaries, or long-range
dependencies.

In this implementation, `embedding_size` must be divisible by `n_heads`:

```text
head_size = embedding_size / n_heads
```

## GPT-style block

Each block uses the pre-layer-normalization pattern:

```text
x = x + multi_head_attention(layer_norm(x))
x = x + feed_forward(layer_norm(x))
```

### Residual connections

The `+` is a residual (or skip) connection. Rather than asking a sub-layer to
replace its input, the sub-layer learns a refinement:

```text
y = x + f(x)
```

This gives information and gradients a direct identity path through the
network. During backpropagation, if the incoming gradient is
`dL / dy`, then:

```text
dL / dx = dL / dy + (dL / dy) (df / dx)
```

The first term comes from the identity branch; the second comes through the
attention or feed-forward branch. Thus, even if the learned branch initially
has weak or poorly conditioned gradients, the identity path still carries a
direct gradient to earlier layers. This makes deep Transformer stacks much
easier to optimize. Some architectures use a learned projection on the skip
path when its input and output dimensions differ; here both have
`embedding_size`, so the identity path needs no projection.

### Layer normalization

`LayerNorm` normalizes each token's embedding features, then applies learned
per-feature scale and shift parameters. Unlike batch normalization, it does
not depend on other examples in the batch, so it works naturally with varying
batch sizes and autoregressive generation.

This implementation is **pre-norm**: normalization happens before attention
and before the feed-forward network. The residual stream itself remains
available unchanged through the skip path:

```text
x -> LayerNorm -> sub-layer --+
|                              +-> x + sub-layer(LayerNorm(x))
`------------------------------+
```

Pre-norm generally improves gradient flow and training stability for deeper
Transformer models.

The feed-forward network applies the same MLP independently to every token
position; attention is the part that lets positions exchange information.

## Parameter count

Let `V` be vocabulary size, `L` sequence length, `d` `embedding_size`, `h`
`n_heads`, and `N` the number of Transformer blocks. This implementation
requires `d` to be divisible by `h`, so each head has size `d / h`.

For one attention head, each of Q, K, and V projects `d` inputs to `d / h`
outputs. Across all heads, the three projections therefore contain
`3 * d * d` weights. The multi-head output projection contributes `d * d`
weights and `d` bias values. Thus, with the default `qkv_bias=False`,
multi-head attention has:

```text
QKV projections:       3d^2
output projection:     d^2 + d
total attention:       4d^2 + d
```

If `qkv_bias=True`, Q, K, and V add another `3d` bias parameters per block.

The feed-forward network expands `d -> 4d -> d`, so it has:

```text
first linear layer:    4d^2 + 4d
second linear layer:   4d^2 + d
total MLP:             8d^2 + 5d
```

Each block also has two LayerNorm layers. Each LayerNorm has a learned scale
and bias of length `d`, adding `4d` parameters per block. In total, one block
contains:

```text
12d^2 + 10d                 # default: qkv_bias=False
12d^2 + 13d                 # when qkv_bias=True
```

The complete language model additionally has token embeddings (`Vd`),
position embeddings (`Ld`), final LayerNorm (`2d`), and the language-model
head (`Vd + V`). Its default total is:

```text
2Vd + Ld + N(12d^2 + 10d) + 2d + V
```

You can always verify the formula against the actual model, including every
enabled bias, with:

```python
sum(parameter.numel() for parameter in model.parameters())
```

The language model adds learned token and positional embeddings, applies a
stack of these blocks, then projects the final embeddings to vocabulary logits.
During generation it repeatedly samples a next token and appends it to the
context.
