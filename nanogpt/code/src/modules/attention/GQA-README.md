# Grouped-Query Attention

Grouped-Query Attention (GQA) is a single knob placed between the two attention
variants already in this project. Instead of every head owning a key/value pair
(MHA) or all heads sharing exactly one (MQA), heads are partitioned into `G`
groups, and each group shares one K/V projection.

```text
G = n_heads   ->  multi-head attention        (../multihead_attention.py)
1 < G < n_heads ->  grouped-query attention
G = 1         ->  multi-query attention       (../multiqueries_attention.py)
```

Queries are never shared. Every head always keeps its own Q projection, in all
three cases. Only K and V are pooled.

Because `MultiHeadAttention` and `MultiQueriesAttention` are the two endpoints,
implementing GQA means the project ends up with one class that subsumes both
rather than three that overlap.

## The problem it solves

MQA was not introduced to save parameters. It was introduced to shrink the
**KV cache** during autoregressive decoding, and understanding why requires
looking at what generation actually costs.

During training, a full sequence is processed at once and attention is
compute-bound: large matrix multiplications keep the GPU's arithmetic units
busy. During generation, one token is produced at a time. To produce token
`t + 1`, the model needs the keys and values of every previous token. Those are
kept in the KV cache rather than recomputed, so each decoding step:

- reads the entire KV cache from memory,
- performs a very small amount of arithmetic against it,
- appends one new K/V entry.

The ratio of arithmetic to memory traffic — **arithmetic intensity** — is
terrible. Decoding is limited by memory bandwidth, not by FLOPs. The GPU
spends most of its time waiting on reads of the KV cache. So the size of that
cache is close to a direct multiplier on generation latency.

Cache size for one sequence is:

```text
bytes = 2 * n_layers * G * head_size * seq_len * bytes_per_element
        ^
        one each for K and V
```

Note what is absent: `n_heads`. The number of *query* heads does not appear.
Only the number of K/V groups does. This is the whole idea — query heads are
cheap to keep, K/V heads are what you pay for at inference time.

For Llama-2-70B (80 layers, 64 heads, `head_size` 128, 4096 context, fp16),
one sequence's cache is:

```text
MHA   (G = 64)   ~10.7 GB
GQA   (G = 8)    ~1.34 GB
MQA   (G = 1)    ~0.17 GB
```

10.7 GB *per concurrent sequence* is what makes plain MHA impractical at that
scale: it caps how many users a server can batch together, which caps
throughput. The move from MHA to GQA is what makes the memory affordable.

## Why not just use MQA

Because MQA overshoots. Collapsing 64 K/V heads into one is a large reduction
in the model's capacity to represent different relationships, and the GQA paper
reports two consequences: measurable quality degradation, and training
instability.

GQA's claim is that the curve is very steep at the MQA end and very flat
elsewhere. Going from 64 groups to 8 costs almost nothing in quality while
capturing 87% of the memory saving. Going from 8 to 1 saves comparatively
little further memory but gives up real quality. `G = 8` is not a coincidence
across model families — it is roughly where that curve flattens, and it also
matches the number of GPUs in a typical tensor-parallel shard, so each device
owns exactly one K/V group.

```text
Llama 2 70B    64 heads, 8 groups
Llama 3 8B     32 heads, 8 groups
Mistral 7B     32 heads, 8 groups
```

## Uptraining

The GQA paper's second contribution is practical: you do not need to train a
GQA model from scratch. An existing MHA checkpoint can be converted.

For each group, the K projections of its member heads are **mean-pooled** into
a single projection, and likewise for V. The model is then briefly trained
further — the paper uses roughly 5% of the original pre-training compute — to
recover the lost quality.

Mean-pooling rather than selecting one head's weights matters. Picking one
member and discarding the rest throws away information and starts the converted
model much further from a good solution; averaging retains a contribution from
every head and lands close enough that a short uptraining run suffices. The
same reasoning that makes averaging sensible for model souping applies here.

This is worth knowing even if you never do it, because it explains why GQA
appeared in already-released model families rather than only in new ones.

## Implementation

The only real question is how a query head finds its group. With `n_heads`
divisible by `G`:

```text
heads_per_group = n_heads / G
group_of_head(i) = i // heads_per_group
```

So with 4 heads and 2 groups, heads 0 and 1 read group 0, heads 2 and 3 read
group 1. Adjacent heads share — which is arbitrary but conventional, and it is
what makes tensor-parallel sharding line up.

Structurally this is a small edit to `../multiqueries_attention.py`: replace
the single `key`/`value` pair with `G` of them, and give each head the K/V
belonging to its group.

```text
self.key   = ModuleList([Linear(embedding_size, head_size) for _ in range(G)])
self.value = ModuleList([Linear(embedding_size, head_size) for _ in range(G)])

forward(x):
    ks = [key_g(x)   for key_g   in self.key]     # G tensors of (B,T,head_size)
    vs = [value_g(x) for value_g in self.value]
    out = cat([
        head(x, k=ks[i // heads_per_group], v=vs[i // heads_per_group])
        for i, head in enumerate(self.heads)
    ], dim=-1)
    return self.dropout(self.proj(out))
```

The existing `CausalAttention(..., shared_kv=True)` head already accepts
injected `k` and `v`, so it needs no change at all. Add an assertion that
`n_heads % G == 0`, since a non-divisible `G` fails later and confusingly.

A batched implementation would instead keep K/V as a single `(B, G, T, hs)`
tensor and expand it across heads. The usual instrument is
`repeat_interleave`, but prefer `expand` where possible: `repeat_interleave`
materializes real copies, whereas `expand` produces a stride-0 view that costs
no memory. Since the copies are read-only here, the view is sufficient, and
avoiding the materialization is precisely the point of the technique. Getting
this backwards is the classic way to write a GQA layer that saves no memory
whatsoever.

## Parameter count

Per block, with `d = embedding_size`, `h = n_heads`, `G` groups, and
`head_size = d / h`:

```text
Q projections:    h * d * (d/h)  =  d^2
K projections:    G * d * (d/h)  =  d^2 * G/h
V projections:    G * d * (d/h)  =  d^2 * G/h
output proj:      d^2 + d
attention total:  d^2 (2 + 2G/h) + d
```

Setting `G = h` recovers `4d^2 + d`, the MHA figure in `../README.md`; setting
`G = 1` gives the MQA layer's cost. Instantiated with this project's
configuration (`d = 64`, `h = 4`, `V = 89`, `L = 32`, `N = 4`):

```text
              attention/block    whole model
G = 4 (MHA)        16,448           212,825
G = 2 (GQA)        12,352           196,441
G = 1 (MQA)        10,304           188,249
```

These are small differences, and that is the honest picture: at `d = 64` the
parameter saving is nearly irrelevant. The technique is about the KV cache,
which this project does not yet have.

## The missing piece: there is no KV cache here

`NgramLanguageModel.generate` in `../../architecture/ngram_lm.py` recomputes the
entire forward pass for the whole context on every single generated token. The
keys and values for tokens already generated are computed from scratch each
step and thrown away.

That means **MQA and GQA currently save you nothing at inference in this
project** beyond a few thousand parameters. The mechanism they optimize does
not exist in the code.

This makes a KV cache the natural companion exercise, and it is the one that
turns these notes into something observable:

1. Add a cache to `generate`: keep `k` and `v` per layer, and on each step
   project only the newest token and append.
2. Measure tokens/second and cache bytes for `G = 4`, `2`, `1`.
3. The ratios should track `G` almost exactly for the cache, and the speedup
   should become visible as the context grows.

At `seq_length = 32` the effect will be small. Raising the context length is
what makes it show up, since cache size is linear in sequence length while the
rest of the per-step cost is roughly constant.

## What GQA does not change

- It does not change the number of query heads, the attention scale, the causal
  mask, or the output shape.
- It does not reduce attention FLOPs. Every query head still attends over every
  past position; the `O(T^2)` work is untouched. GQA reduces *memory* for K/V,
  FlashAttention reduces *memory traffic* for the score matrix, and the two are
  fully complementary — production models use both together.
- It does not help training throughput much. Training is compute-bound and has
  no KV cache; the benefit is concentrated in autoregressive decoding.

## Reading guide

1. Why decoding is memory-bandwidth-bound while training is compute-bound.
2. Why the KV cache formula contains `G` and not `n_heads`.
3. Why quality falls off sharply at `G = 1` but barely at `G = 8`.
4. How mean-pooling plus uptraining converts an existing MHA checkpoint.
5. Then Multi-head Latent Attention (DeepSeek-V2), which attacks the same
   bottleneck by compressing K/V into a low-rank latent vector instead of
   sharing heads — a different answer to the identical question.

## Sources

- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models
  from Multi-Head Checkpoints* ([paper](https://arxiv.org/abs/2305.13245)).
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*
  ([paper](https://arxiv.org/abs/1911.02150)) — the MQA paper GQA generalizes.
- Pope et al., *Efficiently Scaling Transformer Inference*
  ([paper](https://arxiv.org/abs/2211.05102)) — where the memory-bandwidth
  analysis of decoding is laid out most clearly.
- DeepSeek-AI, *DeepSeek-V2* ([paper](https://arxiv.org/abs/2405.04434)) —
  Multi-head Latent Attention, the low-rank alternative.
