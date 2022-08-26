import math

import einops.layers.torch as einops_torch
import torch
from torch import nn


class PositionalEmbedding(nn.Module):

    def __init__(self, embedding_dim, max_length):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super(PositionalEmbedding, self).__init__()

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
            patch_size: int,
            in_channels: int,
            projection_dim: int
    ):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size

        self.patch_extractor = einops_torch.Rearrange("b c (p h) (p w) -> b (h w) (p p c)", p=patch_size)
        self.patch_projector = nn.Linear(patch_size * patch_size * in_channels, projection_dim)

    def forward(self, images):
        assert images.dim() == 4, f"Expected 4D tensor but got: {images.dim()}D"

        patches = self.patch_extractor(images)
        patches = self.patch_projector(patches)

        return patches
