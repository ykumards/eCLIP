from typing import Callable, List, Tuple, Union

import numpy as np
import albumentations as A
import torch
from torch import nn, Tensor
from torchvision import transforms as image_transforms
from torchvision.transforms import InterpolationMode
from PIL import Image


CLIP_DEFAULT_MEAN = [0.485, 0.456, 0.406]
CLIP_DEFAULT_STD = [0.229, 0.224, 0.225]


class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        transformed = self.transform(image=image)["image"]
        return Image.fromarray(transformed)


class CLIPImageTransform(nn.Module):
    """CLIP image transform
    random resized crop (train mode) or resize and center crop, followed by RGB conversion, tensor conversion, and normalization.

    Args:
        image_size (Union[int, Tuple[int]): desired output image size.
        image_interpolation (torchvision.transforms.InterpolationMode):
            Torchvision interpolation mode used during resizing. Defaults to bicubic.
        image_mean (Tuple[float]): mean of images, used for normalization.
        image_std (Tuple[float]): std of images, used for normalization.
        is_train (bool): Whether transform is run in train mode.

    Inputs:
        image (Union[List[Image], Image]): Image or batch of images upon which
            to apply the transform.
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        image_interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        image_mean: Tuple[float, float, float] = CLIP_DEFAULT_MEAN,
        image_std: Tuple[float, float, float] = CLIP_DEFAULT_STD,
        is_train: bool = True,
    ) -> None:
        super().__init__()
        base_transform = []

        if is_train:
            resize_transform = [
                image_transforms.RandomResizedCrop(
                    image_size, scale=(0.5, 1.0), interpolation=image_interpolation
                )
            ]
        else:
            resize_transform = [
                image_transforms.Resize(image_size, interpolation=image_interpolation),
                image_transforms.CenterCrop(image_size),
            ]
        clahe_transform = AlbumentationsTransform(A.CLAHE(p=0.3))

        self.image_transform = image_transforms.Compose(
            resize_transform
            + [
                clahe_transform,
                image_transforms.ToTensor(),
                image_transforms.Normalize(image_mean, image_std),
            ]
        )
        self.mask_transform = image_transforms.Compose(
            resize_transform + [image_transforms.ToTensor()]
        )

    def forward(self, image: Union[List[Image.Image], Image.Image], is_mask=False) -> Tensor:
        transform_fn = self.mask_transform if is_mask else self.image_transform
        if isinstance(image, Image.Image):
            return transform_fn(image)
        image_result = torch.stack([transform_fn(x) for x in image])
        return image_result
