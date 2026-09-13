import torch
import torch.nn as nn
from src.modules.attention.causal_attention import CausalAttention
from src.modules.position_embedding import PositionEmbedding, RoPE


# Source: https://arxiv.org/pdf/2305.13245
# Keynotes: To quote """Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head.
# GQA-G refers to grouped-query with G groups. GQA-1, with a single group and therefore single key and value head, is equivalent to MQA, while GQA-H, with groups equal to number of heads, is equivalent to MHAs"""

class GroupedQueriesAttention(nn.Module):
    """ Grouped Queries Attention"""

    def __init__(self, n_heads: int, 
                 n_groups: int,
                 head_size: int, 
                 embedding_size: int, 
                 seq_length: int,
                 dropout_rate: float,
                 qkv_bias: bool = False,
                 enable_kv_cache: bool = False,
                 shared_kv: bool = True):
        super().__init__()

        assert n_heads % n_groups ==  0
        self.n_groups = n_groups
        self.head_size = head_size
        self.group_size = n_heads // n_groups
        self.n_heads = n_heads
        self.enable_kv_cache = enable_kv_cache
        self.key = nn.Linear(embedding_size, n_groups * head_size, bias=qkv_bias)
        self.value = nn.Linear(embedding_size, n_groups * head_size, bias=qkv_bias)
        self.heads = nn.ModuleList([
                    CausalAttention(
                        head_size,
                        embedding_size,
                        seq_length,
                        dropout_rate,
                        qkv_bias,
                        shared_kv,
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

        k = self.key(x)   # (B,T_new,n_groups*head_size)
        v = self.value(x) # (B,T_new,n_groups*head_size)

        cos = None
        sin = None
        if isinstance(position_embedding, RoPE):
            cos, sin = position_embedding.precompute_rope_frequencies()
            cos = cos[past_length : past_length + T]
            sin = sin[past_length : past_length + T]
            k = k.view(B, T, self.n_groups, self.head_size).transpose(1, 2)   # (B, G, T, hd)
            k = position_embedding.apply_rotary_emb(k, cos, sin)
            k = k.transpose(1, 2).reshape(B, T, self.n_groups * self.head_size)

        if kv_cache_block is not None and not self.training:

            # dim=1 is the time axis. Without it torch.cat defaults to dim=0 (batch).
            k = torch.cat([kv_cache_block['key'], k], dim=1)
            v = torch.cat([kv_cache_block['value'], v], dim=1)


        # One cache per layer, handed back so the caller can thread it to the next step.
        kv_cache_block = {'key': k, 'value': v} if self.enable_kv_cache and not self.training else None

        B, T, C = k.shape
        k = k.view(B, T, self.n_groups, self.head_size)
        v = v.view(B, T, self.n_groups, self.head_size)


        out = torch.cat(
                [head(x, position_embedding, cos, sin, k=k[:,:, i//self.group_size ,:], v=v[:, :, i//self.group_size, :]) for i, head in enumerate(self.heads)],
                dim=-1,
            )
        out = self.dropout(self.proj(out))

        return out, kv_cache_block
