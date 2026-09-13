import torch
import torch.nn as nn
from src.modules.attention.causal_attention import CausalAttention
from src.modules.attention.flash_attention.flash_attention_v1 import FlashAttentionV1
from src.modules.position_embedding import PositionEmbedding, RoPE


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_heads: int, head_size: int, embedding_size: int, seq_length: int, dropout_rate: float, qkv_bias: bool = False, attention_type: str = "causal", tile_size: int = 4):
        super().__init__()
        attention_classes = {
            "causal": CausalAttention,
            "flash": FlashAttentionV1,
        }
        if attention_type not in attention_classes:
            raise ValueError(
                f"Unknown attention_type {attention_type!r}. "
                f"Choose one of {tuple(attention_classes)}."
            )

        attention_class = attention_classes[attention_type]
        attention_kwargs = {"tile_size": tile_size} if attention_type == "flash" else {}
        self.heads = nn.ModuleList([
            attention_class(
                head_size,
                embedding_size,
                seq_length,
                dropout_rate,
                qkv_bias,
                **attention_kwargs,
            )
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, past_length, position_embedding, kv_cache_block=None):
        B, T, C = x.shape
        if isinstance(position_embedding, PositionEmbedding):
            pos_emb = position_embedding(
                torch.arange(past_length, past_length + T)
            ) # (T,C)
            x = x + pos_emb # (B,T,C)

        # Accepted for a uniform Block call signature, but unsupported: every head here
        # owns its own K/V, so caching would need one cache per head. Block rejects it.
        out = torch.cat([h(x, position_embedding) for h in self.heads], dim=-1)

        out = self.dropout(self.proj(out))
        return out, kv_cache_block
