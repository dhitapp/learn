# FlashAttention

FlashAttention is an **exact**, IO-aware implementation of scaled dot-product
attention. It computes the same result as ordinary attention (up to normal
floating-point differences); it is not an attention approximation or a new
model architecture.

This folder is for notes and a future implementation. The current project’s
`CausalAttention` is the clear, conventional formulation of causal attention.

## The problem it solves

For one head, ordinary attention is:

```text
S = Q K^T / sqrt(head_size)  # scores, shape (T, T)
P = softmax(S)               # attention probabilities, shape (T, T)
O = P V                      # output, shape (T, head_size)
```

The `T x T` score/probability matrices become very large for long sequences.
The main practical bottleneck is often not the matrix multiplications, but
writing these intermediate matrices to high-bandwidth GPU memory (HBM) and
reading them back. Standard attention therefore has quadratic intermediate
memory use in sequence length.

FlashAttention keeps small tiles of Q, K, and V in fast on-chip SRAM, computes
partial results there, and avoids materializing the full `S` or `P` matrix in
HBM. It still has `O(T^2)` attention computation, but substantially reduces
memory traffic and attention's intermediate-memory requirement.

## Key idea: exact attention, one block at a time

The essential trick is not merely splitting matrices into blocks. Softmax
couples every score in a query row: a score in a later K block changes the
normalization of scores from earlier blocks. FlashAttention therefore tracks
extra per-row statistics while it processes blocks, then rescales earlier
partial results when needed.

For a Q block and each successive K/V block:

```text
1. Load the Q, K, and V tiles from slow HBM into fast SRAM.
2. Compute the local score tile and apply the causal mask, if needed.
3. Update the running softmax maximum m and normalizer l.
4. Rescale the previous output contribution to the new normalization.
5. Add the current block's appropriately normalized V contribution.
```

When all K/V blocks have been visited, the accumulated output is exactly the
same attention output that a full-matrix softmax would have produced. The
online statistics `m(x)` and `l(x)` make this possible without keeping an
entire score row in memory.

## Tiling and online softmax

FlashAttention splits Q into row blocks and K/V into column blocks. For one Q
block, it visits K/V blocks one at a time. It must nevertheless produce the
same softmax as if all score columns had been present at once.

For each query row it maintains running statistics:

```text
m = running maximum score
l = running sum of exp(score - m)
O = running, normalized output numerator
```

When a new score block arrives, the maximum may increase. Previous `l` and
`O` are rescaled by `exp(old_m - new_m)`, the new block is incorporated, and
the final output is normalized by `l`. This numerically stable **online
softmax** is the key that makes blockwise, exact attention possible.

Conceptually:

```text
load a Q tile
for each K/V tile:
    load K and V into SRAM
    compute this tile's scores
    apply mask and update online-softmax statistics
    update the output accumulator
write only the final output tile to HBM
```

## Causal masking

For a GPT-style decoder, entries that would attend to the future are masked to
negative infinity before the local softmax update. Tiling changes *how* the
calculation is scheduled, not which tokens a query may attend to. The result
therefore preserves the causal behavior in `../causal_attention.py`.

## Backward pass trade-off

Standard implementations often save the full attention matrix for
backpropagation. FlashAttention instead saves much smaller per-row softmax
statistics and recomputes needed score tiles during the backward pass. This
uses extra arithmetic to avoid much more expensive HBM reads and writes—a good
trade-off on modern GPUs.

More concretely, backward needs information derived from the score and
probability matrices, `S` and `P`, to compute gradients for Q, K, and V.
Rather than storing their `T x T` values, FlashAttention saves the final output
`O` and per-row normalization statistics (`m`, `l`). It reloads small Q/K/V
tiles into SRAM and reconstructs the necessary local `S` and `P` values. This
is a selective form of gradient checkpointing: recompute cheap-to-recreate
intermediates, while avoiding costly HBM storage and traffic.

## Kernel fusion

Tiling also lets a GPU implementation fuse multiple steps into one kernel:

```text
read tiles from HBM
  -> matrix multiply for scores
  -> masking / online softmax / optional dropout
  -> matrix multiply with V
  -> write final output tile to HBM
```

Without fusion, these stages often launch separate kernels and repeatedly
write/read large intermediate tensors from HBM. Fusion keeps the temporary
values local to the GPU's fast memory, which is why doing more FLOPs can still
make FlashAttention faster.

## What FlashAttention does not change

- It does not remove the quadratic number of query--key interactions.
- It does not change the parameters, attention scale, causal mask, or model
  outputs in exact arithmetic.
- It does not automatically make a small model more accurate; it makes the
  same attention computation more memory-efficient and often faster.

## Reading guide

While reading the paper, focus on these ideas in order:

1. Why HBM-to-SRAM data movement, rather than FLOPs alone, limits attention.
2. How the online softmax updates preserve an exact result across tiles.
3. How tiling avoids storing the `T x T` attention matrix.
4. Why backward recomputation can reduce total runtime and memory use.

## Source

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness* ([paper](https://arxiv.org/abs/2205.14135)).


