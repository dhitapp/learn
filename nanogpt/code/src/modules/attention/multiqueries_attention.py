import torch
import torch.nn as nn
from src.modules.attention.causal_attention import CausalAttention
from src.modules.attention.flash_attention.flash_attention_v1 import FlashAttentionV1


# Source: https://arxiv.org/pdf/1911.02150
# Keynotes: To quote """Multi-query attention is identical except that the different heads share a single set of keys and values"""

class MultiQueriesAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_heads: int, head_size: int, embedding_size: int, seq_length: int, dropout_rate: float, qkv_bias: bool = False, shared_kv: bool = True):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=qkv_bias)
        self.value = nn.Linear(embedding_size, head_size, bias=qkv_bias)
        self.heads = nn.ModuleList([
                    CausalAttention(
                        head_size,
                        embedding_size,
                        seq_length,
                        dropout_rate,
                        qkv_bias,
                        True
                    )
                    for _ in range(n_heads)
                ])
        
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        # Keynotes: To quote """Multi-query attention is identical except that the different heads share a single set of keys and values"""
        k = self.key(x)   # (B,T,C)
        v = self.value(x) # (B,T,C)

        out = torch.cat(
                [head(x, k=k, v=v) for head in self.heads],
                dim=-1,
            ) 
        out = self.dropout(self.proj(out))
        return out
