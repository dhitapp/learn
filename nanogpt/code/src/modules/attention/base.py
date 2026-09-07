import torch
import torch.nn as nn
from torch.nn import functional as F

class Attention(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size: int, embedding_size: int, seq_length: int, dropout_rate: float, 
                 qkv_bias: bool = False,
                 shared_kv: bool = False):
        super().__init__()

        self.head_size = head_size
        self.shared_kv = shared_kv
        self.query = nn.Linear(embedding_size, head_size, bias=qkv_bias)
        if not shared_kv:
            self.key = nn.Linear(embedding_size, head_size, bias=qkv_bias)
            self.value = nn.Linear(embedding_size, head_size, bias=qkv_bias)

        self.register_buffer('tril', torch.tril(torch.ones(seq_length, seq_length)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self):
        pass