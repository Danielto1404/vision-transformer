from torch import nn

from vit.layers.attention import MultiHeadAttention


class PreLayerNorm(nn.Module):
    def __init__(
            self,
            dim: int,
            layer: nn.Module
    ):
        super(PreLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.layer = layer

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.layer(x, **kwargs)

        return x


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            dropout=0.0
    ):
        super(FeedForward, self).__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768,
            heads: int = 12,
            head_dim: int = 64,
            feedforward_dim: int = 2048,
            dropout: float = 0.0
    ):
        super(TransformerEncoderLayer, self).__init__()

        mha = MultiHeadAttention(
            embedding_dim=embedding_dim,
            heads=heads,
            head_dim=head_dim,
            dropout=dropout
        )

        mlp = FeedForward(embedding_dim, feedforward_dim, dropout)

        self.normed_mha = PreLayerNorm(embedding_dim, mha)
        self.normed_mlp = PreLayerNorm(embedding_dim, mlp)

    def forward(self, x):
        x = self.normed_mha(x) + x
        x = self.normed_mlp(x) + x

        return x


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            layers: int,
            embedding_dim: int = 768,
            heads: int = 12,
            head_dim: int = 64,
            feedforward_dim: int = 2048,
            dropout: float = 0.0
    ):
        super(TransformerEncoder, self).__init__()

        self.layers = layers

        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(
                embedding_dim=embedding_dim,
                heads=heads,
                head_dim=head_dim,
                feedforward_dim=feedforward_dim,
                dropout=dropout
            ) for _ in range(layers)
        ])

    def forward(self, x):
        for layer in self.encoders:
            x = layer(x)

        return x
