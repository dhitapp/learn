import torch
import torch.nn as nn
from torch.nn import functional as F

class Attention(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size: int, embedding_size: int, seq_length: int, dropout_rate: float, qkv_bias: bool = False):
        super().__init__()
        # these 3 lines of codes can also be represented as self.qkv_proj = nn.Linear(embedding_size, 3*head_size, bias=bias)
        self.key = nn.Linear(embedding_size, head_size, bias=qkv_bias)
        self.query = nn.Linear(embedding_size, head_size, bias=qkv_bias)
        self.value = nn.Linear(embedding_size, head_size, bias=qkv_bias)

        self.register_buffer('tril', torch.tril(torch.ones(seq_length, seq_length)))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self):
        pass