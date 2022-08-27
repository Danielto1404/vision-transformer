## Vision Transformer 

This repository contains [PyTorch](https://pytorch.org/) implementation of original
paper: *[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)*

"While the _Transformer_ architecture has become the de-facto standard for natural language processing tasks, its
applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional
networks, or used to replace certain components of convolutional networks while keeping their overall structure in
place. We show that this reliance on _CNNs_ is not necessary and a pure transformer applied directly to sequences of
image
patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred
to multiple mid-sized or small image recognition benchmarks _(ImageNet, CIFAR-100, VTAB, etc.)_, _Vision Transformer (
ViT)_
attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer
computational resources to train."

### Installation

_HTTPS_:

```shell
$ pip install git+https://github.com/Danielto1404/vision-transformer.git
```

_SSH_:

```shell
$ pip install git@github.com:Danielto1404/vision-transformer.git
```

### VIT Feature extractor example

```python
import torch
from vit import VIT

model = VIT(
    image_size=(3, 28, 28),    # channels x height x width  
    patch_size=4,              # n x n patch
    embedding_dim=768,         # embedding dimension which
    layers=4,                  # number of transformer encoder layers
    heads=12,                  # number of transformer encoder heads
    head_dim=64,               # single head dimension
    feedforward_dim=2048,      # transformer encoder mlp dimension
    dropout=0.2,               # dropout
    pooling="cls"              # [`cls`, `mean`]
)

x = torch.rand(32, 3, 28, 28)  # batch x channels x height x width
features = model(x)            # batch x embedding_dim
```

### Classifier Example

```python
import torch
from vit import VITClassifier

model = VITClassifier(
    num_classes=10,            # number of classes
    image_size=(3, 28, 28),    # channels x height x width  
    patch_size=14,             # n x n patch
    embedding_dim=768,         # embedding dimension
    layers=4,                  # number of transformer encoder layers
    heads=12,                  # number of transformer encoder heads
    head_dim=64,               # single head dimension
    feedforward_dim=2048,      # transformer encoder mlp dimension
    dropout=0.2,               # dropout
    pooling="cls"              # [`cls`, `mean`]
)

x = torch.rand(32, 3, 28, 28)  # batch x channels x height x width
classes = model(x)             # batch x num_classes (32 x 10)
```

### Pretrained models

There are a few pretrained models:
    _TODO_