from typing import Tuple

import torch
import torch.nn as nn

from layers.embeddings import PatchEmbedding
from layers.poolings import ClsPooler, MeanPooler
from layers.transformer import TransformerEncoder


class ViT(nn.Module):
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
        super(ViT, self).__init__()

        channels, height, width = image_size

        assert pooling in ["cls", "mean"], "Unknown pooling type, possible pooling: [`cls`, `mean`]"

        assert height % patch_size == 0 and width % patch_size == 0, \
            "Image dimensions must be divisible by the patch size."

        num_patches = (height // patch_size) * (width // patch_size)

        self.pooling = pooling

        self.embedding = PatchEmbedding(patch_size, channels, embedding_dim)
        self.transformer = TransformerEncoder(
            layers=layers,
            embedding_dim=embedding_dim,
            heads=heads,
            head_dim=head_dim,
            feedforward_dim=feedforward_dim,
            dropout=dropout
        )

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.rand(embedding_dim))
            self.pos_embed = nn.Parameter(torch.randn(num_patches + 1, embedding_dim))
            self.pooler = ClsPooler()
        else:
            self.cls_token = None
            self.pos_embed = nn.Parameter(torch.randn(num_patches, embedding_dim))
            self.pooler = MeanPooler()

    def forward(self, x):
        x = self.embedding(x)

        # Add CLS Token
        if self.pooling_type == "cls":
            b = x.size(0)
            c = self.cls_token.expand(b, 1, -1)
            x = torch.cat([c, x], 1)

        x = self.pos_embed + x
        x = self.transformer(x)
        x = self.pooler(x)

        return x


class ViTForClassification(ViT):
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
        super(ViTForClassification, self).__init__(
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
        x = super().forward(x)
        x = self.classifier(x)

        return x


__all__ = [
    "ViT",
    "ViTForClassification"
]
