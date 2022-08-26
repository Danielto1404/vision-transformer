## Vision Transformer

This repository contains implementation of original
paper: *[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)*

"While the _Transformer_ architecture has become the de-facto standard for natural language processing tasks, its
applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional
networks, or used to replace certain components of convolutional networks while keeping their overall structure in
place. We show that this reliance on _CNNs_ is not necessary and a pure transformer applied directly to sequences of image
patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred
to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), _Vision Transformer (ViT)_
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

# model = VIT(
#     image_size=(3, 28, 28),  # channels x height x width  
#     patch_size=4,  # n x n patch
#     layers=4,  # transformer layers
#     heads=4,  # number od transformer encoder heads
#     feedforward_dim=64,  # transformer encoder mlp dimension
#     dropout=0.5  # dropout
# )
# 
# d_model = model.d_model  # 48 (patch * patch * channels)
# 
# x = torch.rand(32, 3, 28, 28)  # batch x channels x height x width
# features = model(x)  # batch x d_model
```

### Classifier Example

```python
# import torch
# from vit import VITClassifier
# 
# classifier = VITClassifier(
#     image_size=(3, 28, 28),  # channels x height x width 
#     num_classes=10,  # amount of classes 
#     patch_size=4,  # n x n patch
#     layers=4,  # transformer layers
#     heads=4,
#     # number od transformer encoder heads
#     feedforward_dim=64,  # transformer encoder mlp dimension
#     dropout=0.5  # dropout
# )
# 
# x = torch.rand(32, 3, 28, 28)  # batch x channels x height x width
# classes = classifier(x)  # batch x num_classes
```


### Pretrained models

There are a few pretrained models:
* [ImageNet weights]()
* [ImageNet weights]()