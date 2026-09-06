import torch
import torch.nn as nn
from torch.nn import functional as F
from src.modules.attention.base import Attention
from src.modules.attention.causal_attention import CausalAttention


class MultiHeadAttention(Attention):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_heads: int, head_size: int, embedding_size: int, seq_length: int, dropout_rate: float, qkv_bias: bool = False):
        super().__init__(head_size, embedding_size, seq_length, dropout_rate, qkv_bias)
        self.heads = nn.ModuleList([CausalAttention(head_size, embedding_size, seq_length, dropout_rate, qkv_bias) for _ in range(n_heads)])
        self.proj = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out