from typing import Tuple

import torch
from torch import nn

from layers.embeddings import PatchEmbedding, SinCosEmbedding
from layers.poolings import ClsPooling, MeanPooling
from layers.transformer import TransformerEncoder


class VIT(nn.Module):
    def __init__(
            self,
            image_size: Tuple[int, int, int],
            patch_size: int,
            embedding_dim: int = 768,
            layers: int = 4,
            heads: int = 12,
            head_dim: int = 64,
            feedforward_dim: int = 2048,
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
        self.pooling_type = pooling

        self.patch_embedding = PatchEmbedding(patch_size, channels, embedding_dim)

        self.transformer = TransformerEncoder(
            layers=layers,
            embedding_dim=embedding_dim,
            heads=heads,
            head_dim=head_dim,
            feedforward_dim=feedforward_dim,
            dropout=dropout
        )

        if self.pooling_type == "cls":
            self.cls_token = nn.Parameter(torch.rand(self.embedding_dim))
            self.pos_embed = nn.Parameter(torch.randn(self.num_patches + 1, self.embedding_dim))
            self.pooling = ClsPooling()
        else:
            self.cls_token = None
            self.pos_embed = nn.Parameter(torch.randn(self.num_patches, self.embedding_dim))
            self.pooling = MeanPooling()

    def forward(self, x):
        x = self.patch_embedding(x)

        # Add CLS Token
        if self.pooling_type == "cls":
            b = x.size(0)
            c = self.cls_token.expand(b, 1, -1)
            x = torch.cat([c, x], 1)

        x = self.pos_embed + x
        x = self.transformer(x)
        x = self.pooling(x)

        return x


class VITClassifier(VIT):
    def __init__(
            self,
            num_classes: int,
            image_size: Tuple[int, int, int],
            patch_size: int,
            embedding_dim: int = 768,
            layers: int = 4,
            heads: int = 12,
            head_dim: int = 64,
            feedforward_dim: int = 2048,
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
