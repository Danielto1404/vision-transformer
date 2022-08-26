from typing import Tuple

import torch
from torch import nn

from layers.embeddings import PatchEmbedding, PositionalEmbedding
from layers.poolings import ClsPooling, MeanPooling
from layers.transformer import TransformerEncoder


class VIT(nn.Module):
    def __init__(
            self,
            image_size: Tuple[int, int, int],
            patch_size: int,
            embedding_dim: int,
            layers: int = 4,
            heads: int = 8,
            head_dim: int = 64,
            feedforward_dim: int = 1024,
            dropout: float = 0.0,
            pooling: str = "cls"
    ):
        super(VIT, self).__init__()

        channels, height, width = image_size

        assert pooling in ["cls", "mean"], "Unknown pooling type, possible pooling: [`cls`, `mean`]"

        assert height % patch_size == 0 and width % patch_size == 0, \
            "Image dimensions must be divisible by the patch size."

        self.channels = channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = (height // patch_size) * (width // patch_size)
        self.patch_dim = channels * patch_size * patch_size

        self.cls_token = nn.Parameter(torch.rand(1, 1, self.embedding_dim))

        self.positional_embedding = PositionalEmbedding(
            embedding_dim=embedding_dim,
            max_length=self.num_patches + 1  # add extra cls token
        )

        self.patch_embedding = PatchEmbedding(patch_size, channels, embedding_dim)

        self.transformer = TransformerEncoder(
            embedding_dim=embedding_dim,
            layers=layers,
            heads=heads,
            head_dim=head_dim,
            feedforward_dim=feedforward_dim,
            dropout=dropout
        )

        self.pooling = ClsPooling() if pooling == "cls" else MeanPooling()

    def forward(self, x):
        batch, _, _, _ = x.shape

        x = self.patch_embedding(x)

        cls = self.cls_token.expand(batch, 1, -1)
        x = torch.cat([cls, x], 1)

        x = self.positional_embedding(x)
        x = self.transformer(x)
        x = self.pooling(x)

        return x


class VITClassifier(VIT):
    def __init__(
            self,
            num_classes: int,
            image_size: Tuple[int, int, int],
            patch_size: int,
            embedding_dim: int,
            layers: int = 4,
            heads: int = 8,
            head_dim: int = 64,
            feedforward_dim: int = 1024,
            dropout: float = 0.0,
            pooling: str = "cls"
    ):
        super(VITClassifier, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            layers=layers,
            heads=heads,
            head_dim=head_dim,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            pooling=pooling
        )

        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        x = super(VITClassifier, self).forward(x)
        x = self.classifier(x)

        return x
