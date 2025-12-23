"""
Composable Transforms - Individual augmentation operations.

Each transform wraps an Albumentations transform and provides a
consistent interface. All transforms inherit from BaseTransform.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Any

import albumentations as A


@dataclass
class TransformResult:
    """Result from applying a transform."""
    image: Any  # numpy array
    bboxes: List[List[float]]
    class_labels: List[int]


class BaseTransform(ABC):
    """Base class for all transforms."""

    @abstractmethod
    def get_albumentations_transform(self) -> A.BasicTransform:
        """Return the underlying Albumentations transform."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Transform name for logging."""
        pass


class FlipTransform(BaseTransform):
    """Horizontal and vertical flip transforms."""

    def __init__(self, horizontal_p: float = 0.5, vertical_p: float = 0.0):
        self.horizontal_p = horizontal_p
        self.vertical_p = vertical_p

    @property
    def name(self) -> str:
        return "flip"

    def get_albumentations_transform(self) -> A.Compose:
        transforms = []
        if self.horizontal_p > 0:
            transforms.append(A.HorizontalFlip(p=self.horizontal_p))
        if self.vertical_p > 0:
            transforms.append(A.VerticalFlip(p=self.vertical_p))
        return A.Compose(transforms) if transforms else A.NoOp()


class RotateTransform(BaseTransform):
    """Rotation transform."""

    def __init__(self, limit: int = 15, p: float = 0.3):
        self.limit = limit
        self.p = p

    @property
    def name(self) -> str:
        return "rotate"

    def get_albumentations_transform(self) -> A.Rotate:
        return A.Rotate(limit=self.limit, p=self.p, border_mode=0)


class ColorTransform(BaseTransform):
    """Brightness, contrast, and color adjustments."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        hue: int = 20,
        saturation: int = 30,
        value: int = 20,
        p: float = 0.3
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.p = p

    @property
    def name(self) -> str:
        return "color"

    def get_albumentations_transform(self) -> A.Compose:
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=self.brightness,
                contrast_limit=self.contrast,
                p=self.p
            ),
            A.HueSaturationValue(
                hue_shift_limit=self.hue,
                sat_shift_limit=self.saturation,
                val_shift_limit=self.value,
                p=self.p
            ),
        ])


class NoiseTransform(BaseTransform):
    """Blur and noise augmentations."""

    def __init__(
        self,
        blur_limit: int = 3,
        blur_p: float = 0.1,
        noise_std: Tuple[float, float] = (0.05, 0.15),
        noise_p: float = 0.1
    ):
        self.blur_limit = blur_limit
        self.blur_p = blur_p
        self.noise_std = noise_std
        self.noise_p = noise_p

    @property
    def name(self) -> str:
        return "noise"

    def get_albumentations_transform(self) -> A.Compose:
        return A.Compose([
            A.GaussianBlur(blur_limit=self.blur_limit, p=self.blur_p),
            A.GaussNoise(std_range=self.noise_std, p=self.noise_p),
        ])


class ScaleTransform(BaseTransform):
    """Random scaling/resizing."""

    def __init__(self, scale_limit: float = 0.1, p: float = 0.3):
        self.scale_limit = scale_limit
        self.p = p

    @property
    def name(self) -> str:
        return "scale"

    def get_albumentations_transform(self) -> A.RandomScale:
        return A.RandomScale(scale_limit=self.scale_limit, p=self.p)


class CropTransform(BaseTransform):
    """Random cropping."""

    def __init__(self, height: int = 512, width: int = 512, p: float = 0.3):
        self.height = height
        self.width = width
        self.p = p

    @property
    def name(self) -> str:
        return "crop"

    def get_albumentations_transform(self) -> A.RandomCrop:
        return A.RandomCrop(height=self.height, width=self.width, p=self.p)
