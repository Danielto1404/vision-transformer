from torch import nn


class ImagePatcher(nn.Module):
    """
    The standard Transformer receives as input a `1D` sequence of token embeddings.
    To handle `2D` images, we reshape the image
        .. math:: X^{W \cdot H \cdot C}  X^{N \cdot (P \cdot P \cdot C)}
    """

    def __init__(self, patch_size: int):
        super(ImagePatcher, self).__init__()
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


if __name__ == '__main__':
    import torchvision.io
    import matplotlib.pyplot as plt

    img = torchvision.io.read_image('../assets/tukan.jpeg')

    PATCH_SIZE = 160
    CHANNELS = img.shape[0]

    patcher = ImagePatcher(PATCH_SIZE)

    patch, (c, h, w) = patcher(img.unsqueeze(0))
    patch = patch[0]

    fig, axs = plt.subplots(h, w, sharex='all', sharey='all')

    for hi in range(h):
        for wi in range(w):
            ax = axs[hi][wi]

            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(patch[hi * w + wi].reshape(CHANNELS, PATCH_SIZE, PATCH_SIZE).permute(1, 2, 0).numpy())

    plt.subplots_adjust(wspace=0.1, hspace=0.05)
    plt.show()
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.show()
