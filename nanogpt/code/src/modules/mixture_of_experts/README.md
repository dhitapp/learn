# Mixture of Experts

A Mixture of Experts (MoE) layer replaces one dense sub-layer with several
parallel copies of it, plus a small **router** that sends each token to only a
few of them. The layer therefore holds many more parameters than the dense
version, while each individual token still performs roughly the same amount of
arithmetic.

This is **conditional computation**: the parameter count and the per-token
compute cost stop being the same number. It is not an approximation of the
dense layer, and it is not a change to attention.

## What gets replaced, and why

In a GPT block, the position-wise feed-forward network is the natural target.
Recall the accounting in `../attention/README.md`, with `d = embedding_size`:

```text
attention:    4d^2 + d
feed-forward: 8d^2 + 5d
2 LayerNorms: 4d
block total:  12d^2 + 10d
```

The feed-forward network is about two thirds of every block. It is also applied
to each token position independently — attention is the part that moves
information between positions. A sub-layer that already treats tokens
independently can be sharded per token without disturbing the sequence
structure at all. Attention cannot: tokens must still be able to see each
other's keys and values.

So in this project the MoE would replace `FeedFoward` inside `Block`
(`../architecture/ngram_lm.py`):

```text
x = x + self.sa(self.ln1(x))      # unchanged
x = x + self.moe(self.ln2(x))     # was self.ffwd
```

## The router

Let `E` be the number of experts and `k` the number activated per token. The
router is a single linear map from the token's embedding to one logit per
expert:

```text
logits = x @ W_router               # (B, T, E), typically no bias
top_val, top_idx = topk(logits, k)  # (B, T, k)
gates = softmax(top_val)            # renormalized over the k winners only
y = sum over j of gates[j] * expert[top_idx[j]](x)
```

Two details in that snippet matter more than they look.

**The softmax comes after the top-k, not before.** Taking the top `k` of a
full softmax and then using those probabilities leaves the gates summing to
less than one, and by a different amount for every token. Renormalizing over
only the selected experts keeps the layer's output scale stable regardless of
how confident the router was.

**The gate value must multiply the expert output.** This is the single most
important line in an MoE. `topk` is a discrete selection: it has no useful
gradient with respect to `W_router`. The only path by which the router ever
learns is `gates[j]`, a differentiable scalar sitting in front of the expert's
output. If you were to select experts with the router and then just sum their
raw outputs, `W_router` would receive exactly zero gradient and the routing
would stay frozen at its initialization forever. The gradient the router
receives is

```text
dL/d gates[j] = <dL/dy, expert_j(x)>
```

which reads as: *increase the weight on this expert if its output happened to
point in the direction that reduced the loss.* The router is trained purely by
this credit assignment, never by a routing label.

### Noisy top-k gating

Shazeer et al. add tunable Gaussian noise to the logits before selection:

```text
H(x)_i = (x @ W_router)_i + StandardNormal() * softplus((x @ W_noise)_i)
```

The noise is exploration. Without it the router commits early to whichever
experts happened to start with slightly favorable weights, and never discovers
that another expert would have done better. `W_noise` is learned, so the model
can decide per-token how much exploration it wants, and can anneal it away.
This is applied during training only.

## Load balancing

MoE has one dominant failure mode, and every production system is built around
preventing it.

Routing is self-reinforcing. An expert that receives slightly more tokens gets
more gradient updates, becomes better, and so earns a higher router logit,
which brings it still more tokens. Left alone, the router collapses onto a
small subset of experts. The rest are never selected, never trained, and the
layer degenerates into a dense FFN that is `E` times more expensive to store.
This is **expert collapse**, and it happens quickly and quietly — the loss
curve can look entirely normal while three quarters of your parameters are
dead.

The standard countermeasure is an auxiliary loss added to the cross-entropy.
In the Switch Transformer formulation, over a batch:

```text
f_i = fraction of tokens that were dispatched to expert i
P_i = mean router probability assigned to expert i
L_aux = alpha * E * sum over i of (f_i * P_i)
```

`f_i` comes from counting, so it is not differentiable; `P_i` is, and it is the
term the gradient actually flows through. The product is minimized when both
are uniform. The factor `E` normalizes the value: under perfectly uniform
routing each term is `(1/E)(1/E)`, summing to `1/E`, so `L_aux` lands at `1.0`
independent of how many experts you chose. `alpha` is typically `0.01` — large
enough to keep experts alive, small enough not to fight the language-modeling
objective.

Two further mechanisms appear in larger systems:

- **Capacity factor.** Each expert is given a fixed buffer,
  `capacity = (tokens_per_batch / E) * capacity_factor`, so the dispatch can be
  a fixed-shape tensor. Tokens overflowing that buffer are *dropped* — they
  skip the layer and reach the next one through the residual connection alone.
  This is a concession to distributed execution, where experts live on
  different devices and shapes must be known ahead of time. A single-GPU
  implementation like this project's does not need it.
- **Router z-loss.** `mean((logsumexp(router_logits))^2)`, which penalizes
  router logits from growing large. Introduced in ST-MoE to fix instability
  and divergence in low precision.

## Top-1 versus top-2

`k` sets what you are buying.

- `k = 1` (Switch Transformer): per-token FFN compute is identical to the dense
  model, so you get `E` times the parameters essentially for free in FLOPs. The
  routing decision is brittle, though — there is no second opinion — so it
  leans harder on the auxiliary loss and on noise.
- `k = 2` (GShard, Mixtral): costs **two** dense FFNs per token, so the layer
  is roughly 2x the dense FLOPs. In exchange the gates form a genuine weighted
  combination, gradients are less noisy, and training is markedly more stable.

Mixtral 8x7B is the familiar reference point: `E = 8`, `k = 2`, about 47B total
parameters of which roughly 13B are active for any given token.

## Parameter and compute accounting

For one block, with `E` experts and no `qkv_bias`:

```text
attention:        4d^2 + d
router:           dE
experts:          E * (8d^2 + 5d)
2 LayerNorms:     4d
block total:      4d^2 + d + dE + E(8d^2 + 5d) + 4d
```

Instantiated with this project's configuration — `d = 64`, `V = 89`,
`L = 32`, `N = 4` blocks — and `E = 4`:

```text
                        dense        MoE E=4
per-block FFN params    33,088       132,608 (incl. 256-param router)
per-block total         49,792       149,312
whole model            212,825       610,905
```

That is 2.87x the parameters of the model in `train.ipynb`. The active
per-token cost, however, depends only on `k`:

```text
                        active FFN params per token
dense                   33,088
MoE E=4, k=1            33,088   + 256 router   (+0.5% per block)
MoE E=4, k=2            66,176   + 256 router   (~2x the dense FFN)
```

The whole point of the design is visible in the gap between those two tables:
storage scales with `E`, arithmetic scales with `k`.

You can always check the formula against reality with:

```python
sum(parameter.numel() for parameter in model.parameters())
```

## Implementation shape

The naive version — loop over experts, and give each one the tokens that chose
it — is the right one to write first. It is easy to verify and it is what a
character-level model on one GPU should use:

```text
flatten x to (B*T, d)
compute router logits, top-k indices, renormalized gates
allocate output = zeros_like(flat_x)
for each expert i:
    mask = positions where i appears among that token's top-k
    if mask is empty: continue
    output[mask] += gate_for_i[mask] * expert_i(flat_x[mask])
reshape output back to (B, T, d)
```

The `continue` on an empty mask is not an optimization; passing a zero-length
batch through `nn.Linear` is fine, but the guard keeps the intent obvious and
avoids surprises with normalization layers.

Note what this loop already gives you: each expert sees a contiguous batch of
only its own tokens, so the FLOPs really are sparse. A tempting shortcut is to
run every expert on every token and then mask the outputs; that is much simpler
to write, produces numerically identical results, and is **`E` times more
expensive than dense** — it throws away the entire benefit. It is worth
building once to test against, and worth never shipping.

### Integration points in this project

Two things in the existing code will need to change, and both are worth
noticing before starting:

1. `self.blocks` is an `nn.Sequential` in `../architecture/ngram_lm.py`.
   `nn.Sequential` passes exactly one tensor from block to block, with nowhere
   to put a per-block auxiliary loss. It becomes an `nn.ModuleList` with an
   explicit `for` loop, or each block stashes its aux loss on itself
   (`self.last_aux_loss`) to be gathered afterwards.
2. `NgramLanguageModel.forward` returns `(logits, loss)`. The auxiliary losses
   from every block have to be summed into that `loss` before it is returned,
   or the load-balancing term never reaches `loss.backward()` and the experts
   collapse exactly as if it had never been written.

The auxiliary loss should be computed but **excluded from the number you
report** as train/val loss — otherwise the curves in the notebook are no longer
comparable with the dense runs already recorded there.

## What to expect at this scale

Honestly: probably not an improvement. This project trains a 0.2M-parameter
character-level model on about 1.26 MB of text, and `train.ipynb` already ends
at train 1.31 against val 1.70 — a gap that says the dense model is memorizing
the corpus. Tripling the parameter count will widen that gap, not close it. MoE
pays off when a model is capacity-limited and data is plentiful, which is the
opposite of the situation here.

That is a reason to implement it, not a reason to skip it. The mechanisms worth
learning — the discrete-selection gradient path, collapse and the balancing
loss, the parameter/FLOP split — are all fully visible at this scale, and they
are much easier to instrument in a model that trains in minutes. Just measure
the right things:

- **Expert utilization histogram**, tokens per expert per batch. This is the
  first plot to make and the one that tells you whether anything is working.
- **Router entropy** over time. Collapsing to near-zero means the router has
  saturated and stopped exploring.
- **Dead experts**: any expert whose count stays at zero for many steps.
- `L_aux` itself, which should settle near `1.0` if routing is balanced.

Run it with `alpha = 0` once, deliberately, and watch the collapse happen. It
is a much better demonstration of why the auxiliary loss exists than any
description of it.

## What MoE does not change

- It does not change attention, the causal mask, or the residual/pre-norm
  structure.
- It does not reduce total parameter storage — it increases it substantially.
  MoE trades memory for compute, which is the reverse of FlashAttention's
  trade.
- It does not make experts semantically interpretable. Empirically they
  specialize on token-level and syntactic regularities far more than on topics,
  and "the medicine expert" is generally not a thing you will find.
- It does not remove the need for the FFN's non-linearity; each expert is still
  an ordinary `d -> 4d -> d` MLP.

## Reading guide

In this order:

1. Why conditional computation decouples parameter count from per-token FLOPs.
2. How the gate value provides the router's only gradient path.
3. Why routing collapses on its own, and how the auxiliary loss counteracts it.
4. What top-1 buys and what top-2 buys.
5. Only then, the distributed-systems machinery — capacity factors, token
   dropping, expert parallelism — which exists because experts sit on different
   devices, and which a single-GPU implementation can ignore.

## Sources

- Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer* ([paper](https://arxiv.org/abs/1701.06538)) —
  noisy top-k gating, and the origin of the load-balancing problem.
- Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with
  Simple and Efficient Sparsity* ([paper](https://arxiv.org/abs/2101.03961)) —
  top-1 routing, capacity factor, the auxiliary loss used above.
- Lepikhin et al., *GShard* ([paper](https://arxiv.org/abs/2006.16668)) —
  top-2 routing and expert parallelism.
- Zoph et al., *ST-MoE: Designing Stable and Transferable Sparse Expert Models*
  ([paper](https://arxiv.org/abs/2202.08906)) — router z-loss and stability.
- Jiang et al., *Mixtral of Experts* ([paper](https://arxiv.org/abs/2401.04088))
  — a concrete, widely used top-2 configuration.
- Dai et al., *DeepSeekMoE* ([paper](https://arxiv.org/abs/2401.06066)) —
  fine-grained experts plus always-on shared experts.
