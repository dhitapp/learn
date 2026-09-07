import torch

from src.modules.attention.base import Attention


class FlashAttentionV1(Attention):
    """One causal attention head using FlashAttention v1's online softmax.

    This is a readable PyTorch reproduction of the paper's forward algorithm.
    It tiles the sequence dimension, but does not directly manage SRAM or fuse
    GPU kernels like a CUDA/Triton implementation would.
    """

    def __init__(self, head_size: int, embedding_size: int, seq_length: int, dropout_rate: float, qkv_bias: bool = False, tile_size: int = 4):
        # Attention sets up the learned Q/K/V projections, causal-mask buffer,
        # and attention-dropout module shared with CausalAttention.
        super().__init__(head_size, embedding_size, seq_length, dropout_rate, qkv_bias)
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        self.head_size = head_size
        self.tile_size = tile_size

    def forward(self, x):
        # T is the current input length, which can be smaller than seq_length.
        _, T, _ = x.shape
        # Project embeddings to one head's query, key, and value vectors:
        # each tensor has shape (B, T, head_size).
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Split token positions into [start:end] tiles. The final tile may be
        # shorter than tile_size when T is not evenly divisible.
        ranges = [(start, min(start + self.tile_size, T)) for start in range(0, T, self.tile_size)]

        # Algorithm 1, line 2: each Q tile keeps its running output O_i,
        # softmax denominator l_i, and row maximum m_i. These states must live
        # across every K/V tile, so they are initialized before the outer loop.
        out_blocks = [torch.zeros_like(q[:, start:end, :]) for start, end in ranges]
        sum_blocks = [torch.zeros(q.shape[0], end - start, device=q.device, dtype=q.dtype) for start, end in ranges]
        max_blocks = [torch.full((q.shape[0], end - start), -torch.inf, device=q.device, dtype=q.dtype) for start, end in ranges]

        # Algorithm 1, lines 5-6: stream one K/V tile at a time. In an actual
        # FlashAttention kernel this is where the tile is loaded into SRAM.
        for kv_start, kv_end in ranges:
            k_block = k[:, kv_start:kv_end, :]
            v_block = v[:, kv_start:kv_end, :]

            # Algorithm 1, lines 7-13: update every Q tile with this K/V tile.
            for index, (q_start, q_end) in enumerate(ranges):
                q_block = q[:, q_start:q_end, :]
                # Local score tile S_ij, shape (B, Br, Bc). Scaling uses the
                # per-head dimension because Q/K dot products have that width.
                scores = q_block @ k_block.transpose(-2, -1) * self.head_size**-0.5

                # Select the corresponding part of the full causal mask. This
                # prevents a query position from attending to future K tokens.
                mask = self.tril[q_start:q_end, kv_start:kv_end].bool()
                scores = scores.masked_fill(~mask, -torch.inf)

                # Algorithm 1, line 10: calculate this tile's stable local
                # softmax state: m_tilde, P_tilde, and l_tilde. A fully masked
                # tile has no valid score; safe_block_max prevents -inf - -inf
                # from producing NaN before P_tilde is set to zero.
                has_valid_score = mask.any(dim=-1).unsqueeze(0)
                block_max = scores.max(dim=-1).values
                safe_block_max = torch.where(has_valid_score, block_max, torch.zeros_like(block_max))
                p_tilde = torch.where(mask.unsqueeze(0), torch.exp(scores - safe_block_max.unsqueeze(-1)), torch.zeros_like(scores))
                block_sum = p_tilde.sum(dim=-1)

                # This is P_tilde @ V_j: the current tile's unnormalized value
                # contribution. Dropout belongs on attention probabilities.
                weighted_values = self.dropout(p_tilde) @ v_block

                # Read the old O_i, l_i, and m_i associated with this Q tile.
                old_out, old_sum, old_max = out_blocks[index], sum_blocks[index], max_blocks[index]
                block_max = torch.where(has_valid_score, block_max, torch.full_like(block_max, -torch.inf))

                # Algorithm 1, line 11: merge old and new normalizations. The
                # scale factors convert both contributions to the same maximum.
                new_max = torch.maximum(old_max, block_max)
                old_scale = torch.exp(old_max - new_max)
                new_scale = torch.exp(block_max - new_max)
                new_sum = old_scale * old_sum + new_scale * block_sum

                # Algorithm 1, line 12: merge the old normalized output with
                # this tile's value contribution, then normalize by l_new.
                out_blocks[index] = (
                    (old_sum * old_scale).unsqueeze(-1) * old_out
                    + new_scale.unsqueeze(-1) * weighted_values
                ) / new_sum.unsqueeze(-1)
                sum_blocks[index], max_blocks[index] = new_sum, new_max

        # Restore token order by joining the completed Q tiles along T.
        return torch.cat(out_blocks, dim=1)

