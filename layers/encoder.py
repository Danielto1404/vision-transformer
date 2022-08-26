from torch import nn

from layers.attentions import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            heads: int,
            feedforward_dim: int,
            dropout: float = 0.0,
            activation=nn.GELU(),
            layer_norm_eps: float = 1e-5
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.heads = heads
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout
        self.activation = activation

        self.multi_head_attention = MultiHeadAttention(d_model, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model)
        )

    def forward(self, x):
        x = self.norm(self.multi_head_attention(x) + x)
        x = self.norm(x + self.mlp(x))

        return x


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            layers: int,
            heads: int,
            feedforward_dim: int,
            dropout: float = 0.0,
            activation=nn.GELU(),
            layer_norm_eps: float = 1e-5
    ):
        super(TransformerEncoder, self).__init__()

        self.layers = layers
        self.heads = heads
        self.feedforward_dim: int

        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                heads=heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps
            ) for _ in range(layers)
        ])

    def forward(self, x):
        for layer in self.encoders:
            x = layer(x)

        return x
