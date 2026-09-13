
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.modules.attention import MultiHeadAttention, MultiQueriesAttention, GroupedQueriesAttention
from src.modules.position_embedding import PositionEmbedding, RoPE

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, embedding_size:int, dropout_rate: float):
        super().__init__()
        self.net = nn.Sequential(

            # we use 4 to follow the original paper (regarding the inner layer)
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_heads: int, 
                 head_size: int,
                 n_groups: int,
                 embedding_size: int, 
                 seq_length: int, 
                 dropout_rate: float, 
                 qkv_bias: bool = False, 
                 attention_type: str = "causal",
                 multi_attention_type: str = 'mha',
                 tile_size: int = 4,
                 enable_kv_cache: bool = False,
                 ):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.enable_kv_cache = enable_kv_cache
        self.seq_length = seq_length
        if multi_attention_type == 'mqa':
            self.sa = MultiQueriesAttention(
                                    n_heads,
                                    head_size,
                                    embedding_size,
                                    seq_length,
                                    dropout_rate,
                                    qkv_bias,
                                    enable_kv_cache,
                                    shared_kv=True
                                )

        elif multi_attention_type == 'gqa':
            self.sa = GroupedQueriesAttention(
                                    n_heads, 
                                    n_groups,
                                    head_size,
                                    embedding_size,
                                    seq_length,
                                    dropout_rate,
                                    qkv_bias,
                                    enable_kv_cache,
                                    shared_kv=True
                                )
        else:
            if enable_kv_cache:
                raise ValueError(
                    "enable_kv_cache is only implemented for multi_attention_type="
                    "'multi-queries'. MultiHeadAttention gives every head its own K/V, "
                    "so it would need one cache per head."
                )
            self.sa = MultiHeadAttention(
                        n_heads,
                        head_size,
                        embedding_size,
                        seq_length,
                        dropout_rate,
                        qkv_bias,
                        attention_type,
                        tile_size,
                    )
        self.ffwd = FeedFoward(embedding_size, dropout_rate)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x, past_length, position_embedding, kv_cache_block=None):
        # `x + self.sa(...)` would try to add a tensor to a tuple: unpack first.
        attention_out, kv_cache_block = self.sa(self.ln1(x), past_length, position_embedding, kv_cache_block)
        x = x + attention_out
        x = x + self.ffwd(self.ln2(x))
        return x, kv_cache_block


# super simple bigram model
class NgramLanguageModel(nn.Module):

    def __init__(self, 
                 vocab_size: int, 
                 n_heads: int,
                 embedding_size: int, 
                 seq_length: int, 
                 n_blocks: int, 
                 dropout_rate: float, 
                 device: str, 
                 n_gram: int = None,
                 enable_kv_cache: bool = False,
                 qkv_bias: bool = False,
                 attention_type: str = "causal",
                 multi_attention_type: str = 'mha',
                 position_embedding_type: str = 'rope',
                 n_groups: int = 0,
                 tile_size: int = 4):
        super().__init__()

        if n_groups < 0:
            raise ValueError("n_groups nneds to be larger than 0")
        
        if multi_attention_type == 'gqa' and n_groups == 0:
            raise ValueError("multi_attention_type='gqa' requires n_groups != 0")

        if multi_attention_type != 'gqa' and n_groups > 0:
            raise ValueError("multi_attention_type='gqa' requires n_groups != 0")
        # each token directly reads off the logits for the next token from a lookup table
        self.head_size = embedding_size // n_heads
        self.enable_kv_cache = enable_kv_cache
        self.seq_length = seq_length
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_type = position_embedding_type
        self.device = device
        if self.position_embedding_type == 'rope':
            self.position_embedding = RoPE(seq_length, self.head_size, self.device)
        else:
            self.position_embedding = PositionEmbedding(seq_length, embedding_size, self.device)
        self.blocks = nn.ModuleList([
            Block(
                n_heads,
                self.head_size,
                n_groups,
                embedding_size,
                seq_length,
                dropout_rate,
                qkv_bias,
                attention_type,
                multi_attention_type,
                tile_size,
                enable_kv_cache
            )
            for _ in range(n_blocks)
        ])
        self.ln_f = nn.LayerNorm(embedding_size) # final layer norm
        self.lm_head = nn.Linear(embedding_size, vocab_size)

        self.device = device

    def forward(self, idx, targets=None, kv_caches=None, return_caches=False):
        B, T = idx.shape

        # How many positions the cache already holds. 
        if kv_caches is not None:
            past_length = kv_caches[-1]['key'].shape[1]
        else:
            past_length = 0

        # idx and targets are both (B,T) tensor of integers
        x = self.token_embedding_table(idx) # (B,T,C)

        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)

        if return_caches:
            ## just in case, to prevent polluted cache
            new_caches = []
        for block, block_cache in zip(self.blocks, kv_caches):
            x, block_cache = block(x, past_length, self.position_embedding, block_cache) # (B,T,C)
            if return_caches:
                new_caches.append(block_cache)

        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        if return_caches:
            return logits, loss, new_caches
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        kv_caches = None
        for _ in range(max_new_tokens):
            if self.enable_kv_cache and kv_caches and idx.shape[1] <= self.seq_length:
                idx_cond = idx[:, -1:]
            else:
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -self.seq_length:]
                kv_caches = None
            # get the predictions
            logits, loss, kv_caches = self(idx_cond, kv_caches=kv_caches, return_caches=True)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
