import torch
import torch.nn as nn
from src.modules.position_embedding.absolute import PositionEmbedding

class RoPE:
    def __init__(self, seq_length: int, dim: int, device:str, base:float = 10000.0):
        super().__init__()
        self.base = base
        self.dim = dim
        self.seq_length = seq_length
        self.device = device
        
    def precompute_rope_frequencies(self):
        """Precompute the frequency tensor for rotary embeddings."""
        # Ensure dimension is even
        assert self.dim % 2 == 0, "Embedding dimension must be even for RoPE."
        
        # Calculate theta values: theta_i = base^(-2(i-1)/dim)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(self.seq_length, dtype=torch.float32)
        
        # Multiply position indices by frequencies (outer product)
        freqs = torch.outer(t, inv_freq) # Shape: (seq_len, dim / 2)
        
        # Concatenate to match full dimension if needed, or keep for complex/halved rotation
        # Emb shape for cos/sin: (seq_len, dim) via repeat_interleave or stacking
        emb = torch.cat([freqs, freqs], dim=-1).to(self.device)
        return torch.cos(emb).to(self.device), torch.sin(emb).to(self.device)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dimensions of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings to input tensor x (Q or K).
        
        Expected shapes:
            x: (batch_size, num_heads, seq_len, head_dim)
            cos, sin: (seq_len, head_dim) -> broadcastable to x
        """
        return (x * cos) + (self.rotate_half(x) * sin)