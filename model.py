import torch
from torch import nn

from layers.encoder import TransformerEncoder
from layers.patches import ImagePatcher


class VIT(nn.Module):
    def __init__(
            self,
            in_channels: int,
            patch_size: int,
            layers: int,
            heads: int,
            feedforward_dim: int = 2048,
            dropout: float = 0.0,
            activation=nn.GELU(),
            layer_norm_eps: float = 1e-5
    ):
        super(VIT, self).__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.d_model = patch_size * patch_size * in_channels

        self.cls_token = torch.zeros(self.d_model, requires_grad=True)

        self.patcher = ImagePatcher(patch_size)
        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            layers=layers,
            heads=heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps
        )

    def forward(self, x):
        # batch x seq x dim
        x = self.patcher(x)

        # batch x (seq + 1) x dim
        x = torch.cat([self.cls_token.unsqueeze(0).unsqueeze(0), x], 1)

        # batch x (seq + 1) x dim
        x = self.encoder(x)

        # batch x dim
        return x[:, 0, :]


class VITClassifier(VIT):
    def __init__(
            self,
            num_classes: int,
            in_channels: int,
            patch_size: int,
            layers: int,
            heads: int,
            feedforward_dim: int = 2048,
            dropout: float = 0.0,
            activation=nn.GELU(),
            layer_norm_eps: float = 1e-5
    ):
        super(VITClassifier, self).__init__(
            in_channels,
            patch_size,
            layers,
            heads,
            feedforward_dim,
            dropout,
            activation,
            layer_norm_eps
        )

        self.num_classes = num_classes

        self.classifier = nn.Linear(
            in_features=self.d_model,
            out_features=num_classes
        )

    def forward(self, x):
        x = super(VITClassifier, self).forward(x)
        x = self.classifier(x)

        return x
