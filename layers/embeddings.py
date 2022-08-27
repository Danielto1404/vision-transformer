import math
from typing import Union, Tuple

from einops import rearrange
import torch
from torch import nn


class SinCosEmbedding(nn.Module):

    def __init__(
            self,
            embedding_dim: int,
            max_length: int
    ):
        """
        Inputs
            embedding_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super(SinCosEmbedding, self).__init__()

        # Create matrix of sequence x embedding representing the positional encoding for max_len inputs
        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class PatchEmbedding(nn.Module):
    def __init__(
            self,
            patch_size: Union[int, Tuple[int, int]],
            in_channels: int,
            embedding_dim: int
    ):
        super(PatchEmbedding, self).__init__()

        self.patch_projector = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, images):
        assert images.dim() == 4, f"Expected 4D tensor but got: {images.dim()}D"

        patches = self.patch_projector(images)
        patches = rearrange(patches, "b c h w -> b (h w) c")

        return patches
