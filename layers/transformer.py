from typing import Union

import torch
from torch import nn

from layers.mha import MultiHeadAttention


class TransformerEncoderMLP(nn.Module):
    def __init__(
            self,
            d_model: int,
            feedforward_dim: int,
            dropout=0.0
    ):
        super(TransformerEncoderMLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            heads: int,
            feedforward_dim: int,
            dropout: float = 0.0,
            layer_norm_eps: float = 1e-5,
            device: Union[str, torch.device] = "cpu"
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.device = device

        self.multi_head_attention = nn.MultiheadAttention(d_model, heads, dropout)

        self.input_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.output_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.mlp = TransformerEncoderMLP(d_model, feedforward_dim, dropout)

    def forward(self, x):
        x = self.input_norm(x)
        a, _ = self.multi_head_attention(x, x, x)
        x = self.output_norm(a + x)
        x = self.mlp(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            layers: int,
            heads: int,
            feedforward_dim: int,
            dropout: float = 0.0,
            layer_norm_eps: float = 1e-5,
            device: Union[str, torch.device] = "cpu"
    ):
        super(TransformerEncoder, self).__init__()

        self.layers = layers
        self.device = device

        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                heads=heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                device=device
            ) for _ in range(layers)
        ])

    def forward(self, x):
        for layer in self.encoders:
            x = layer(x)

        return x
