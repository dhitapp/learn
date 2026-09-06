
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.modules.attention.multihead_attention import MultiHeadAttention

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

    def __init__(self, n_heads: int, embedding_size: int, seq_length: int, dropout_rate: float, qkv_bias: bool = False):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = embedding_size // n_heads
        self.sa = MultiHeadAttention(
                    n_heads,
                    head_size,
                    embedding_size,
                    seq_length,
                    dropout_rate,
                    qkv_bias
                )
        self.ffwd = FeedFoward(embedding_size, dropout_rate)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


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
                 qkv_bias: bool = False):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.seq_length = seq_length
        self.n_gram = n_gram
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(seq_length, embedding_size)
        self.blocks = nn.Sequential(*[Block(n_heads, embedding_size, seq_length, dropout_rate, qkv_bias) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(embedding_size) # final layer norm
        self.lm_head = nn.Linear(embedding_size, vocab_size)

        self.device = device

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.seq_length:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx