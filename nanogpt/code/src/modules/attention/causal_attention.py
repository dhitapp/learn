import torch
import torch.nn as nn
from torch.nn import functional as F
from src.modules.attention.base import Attention
from src.modules.position_embedding import PositionEmbedding, RoPE

class CausalAttention(Attention):
    """ one head of causal-attention """

    def __init__(self, head_size: int, embedding_size: int, seq_length: int, dropout_rate: float, 
                 qkv_bias: bool = False,
                 shared_kv: bool = False):
        super().__init__(head_size, embedding_size, seq_length, dropout_rate, qkv_bias, shared_kv)

    def forward(self, x, position_embedding, cos=None, sin=None, k=None, v=None):
        B, T, C = x.shape
        if self.shared_kv:
            if k is None or v is None:
                raise ValueError("MQA attention requires shared k and v.")
        else:
            k = self.key(x)
            v = self.value(x)
            if isinstance(position_embedding, RoPE):
                cos, sin = position_embedding.precompute_rope_frequencies()
                cos, sin = cos[:T], sin[:T] 
                k = position_embedding.apply_rotary_emb(k, cos, sin)

        q = self.query(x) # (B,T,C)
        if isinstance(position_embedding, RoPE):
            q = position_embedding.apply_rotary_emb(q, cos, sin)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 # (B, Tq, C) @ (B, C, Tk) -> (B, Tq, Tk) (this is scaled)

        # With a KV cache, k holds Tk >= Tq positions: the queries are the *last* Tq
        # rows of the causal mask, not the first Tq. Tk == Tq (no cache) reduces to
        # the usual tril[:T, :T].
        Tk = k.shape[1]
        #we can also use masked_fill_(self.mask[:seq_len, :seq_len].bool(), -torch.inf)
        wei = wei.masked_fill(self.tril[Tk - T:Tk, :Tk] == 0, float('-inf')) # (B, Tq, Tk)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        # perform the weighted aggregation of the values
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out