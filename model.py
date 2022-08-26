from typing import Tuple

import torch
from torch import nn

from layers.transformer import TransformerEncoder
from layers.embeddings import PositionalEncoding, PatchEncoding


class VIT(nn.Module):
    def __init__(
            self,
            image_size: Tuple[int, int, int],
            patch_size: int,
            layers: int,
            heads: int,
            feedforward_dim: int = 2048,
            dropout: float = 0.0,
            layer_norm_eps: float = 1e-5
    ):
        super(VIT, self).__init__()

        channels, height, width = image_size

        self.in_channels = channels
        self.patch_size = patch_size
        self.d_model = patch_size * patch_size * channels
        self.seq_len = height * width // patch_size ** 2

        self.cls_token = torch.zeros(self.d_model, requires_grad=True)

        self.positional_encoder = PositionalEncoding(
            d_model=self.d_model,
            max_len=self.seq_len + 1  # add extra cls token
        )

        self.patch_encoder = PatchEncoding(patch_size)

        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            layers=layers,
            heads=heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps
        )

    def forward(self, x):
        batch, _, _, _ = x.shape

        # batch x seq x dim
        x = self.patch_encoder(x)

        # expand cls token into batch
        cls = self.cls_token.expand(batch, 1, -1)

        # batch x (seq + 1) x dim
        x = torch.cat([cls, x], 1)

        x = self.positional_encoder(x)

        # batch x (seq + 1) x dim
        x = self.encoder(x)

        # batch x dim
        return x[:, 0, :]


class VITClassifier(VIT):
    def __init__(
            self,
            num_classes: int,
            image_size: Tuple[int, int, int],
            patch_size: int,
            layers: int,
            heads: int,
            feedforward_dim: int = 2048,
            dropout: float = 0.0,
            layer_norm_eps: float = 1e-5
    ):
        super(VITClassifier, self).__init__(
            image_size,
            patch_size,
            layers,
            heads,
            feedforward_dim,
            dropout,
            layer_norm_eps
        )

        self.num_classes = num_classes

        self.norm = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.classifier = nn.Linear(
            in_features=self.d_model,
            out_features=num_classes
        )

    def forward(self, x):
        x = super(VITClassifier, self).forward(x)
        x = self.norm(x)
        x = self.classifier(x)

        return x
