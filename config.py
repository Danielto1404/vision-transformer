from typing import Optional, Tuple

from vit import VIT, VITClassifier

VIT_ARCHITECTURES = {
    "ViT-Base-16": dict(
        patch_size=16,
        embedding_dim=768,
        layers=12,
        heads=12,
        head_dim=64,
        feedforward_dim=3072,
    )
}


def vit(name, image_size: Tuple[int, int, int], num_classes: Optional[int]):
    arch = VIT_ARCHITECTURES[name]
    if num_classes is None:
        return VIT(image_size=image_size, **arch)
    else:
        return VITClassifier(num_classes=num_classes, image_size=image_size, **arch)
