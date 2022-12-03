import math

from einops import rearrange
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            heads: int = 8,
            head_dim: int = 64,
            bias: bool = True,
            dropout: float = 0.0
    ):
        super(MultiHeadAttention, self).__init__()

        inner_dim = head_dim * heads

        self.heads = heads
        self.head_dim = head_dim
        self.scale = 1 / math.sqrt(head_dim)

        self.Q_K_V = nn.Linear(embedding_dim, 3 * inner_dim, bias=False)

        self.projector = nn.Sequential(
            nn.Linear(inner_dim, embedding_dim, bias=bias),
            nn.Dropout(dropout)
        )

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        qkv = self.Q_K_V(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "batch seq (heads head_dim) -> batch heads seq head_dim", heads=self.heads), qkv
        )

        dots = q @ k.transpose(-2, -1)
        dots = dots * self.scale

        attentions = self.softmax(dots)
        attentions = self.dropout(attentions)

        out = attentions @ v

        out = rearrange(out, "batch heads seq head_dim -> batch seq (heads head_dim)")

        out = self.projector(out)

        return out


__all__ = [
    "MultiHeadAttention"
]
