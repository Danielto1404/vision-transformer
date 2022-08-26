import math

from torch import nn
import torch


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super(PositionalEncoding, self).__init__()

        # Create matrix of sequence x embedding representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

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


class PatchEncoding(nn.Module):
    """
    The standard Transformer receives as input a `1D` sequence of token embeddings.
    To handle `2D` images, we reshape the image
        .. math:: X^{W \cdot H \cdot C}  X^{N \cdot (P \cdot P \cdot C)}
    """

    def __init__(self, patch_size: int):
        super(PatchEncoding, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        assert x.dim() == 4, f"Expected 4D tensor but got: {x.dim()}D"

        batch, channels, height, width = x.shape

        height_patches = height // self.patch_size
        width_patches = width // self.patch_size

        assert height_patches * self.patch_size == height, "Image height must be divisible by `path_size`"
        assert width_patches * self.patch_size == width, "Image width must be divisible by `path_size`"

        patches = x.unfold(2, self.patch_size, self.patch_size)

        # batch x channels x h_patches x w_patches x patch_size x patch_size
        patches = patches.unfold(3, self.patch_size, self.patch_size)

        # batch x h_patches x w_patches x channels x patch_size x patch_size
        patches = patches.permute(0, 2, 3, 1, 4, 5)

        # batch x (h_patches * w_patches) x (channels * patch_size * patch_size)
        patches = patches.reshape(batch, height_patches * width_patches, -1)

        return patches
