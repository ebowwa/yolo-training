"""
Configuration dataclasses for preprocessing.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations."""
    remove_corrupted: bool = True
    validate_annotations: bool = True
    check_bbox_validity: bool = True
    min_bbox_size: int = 1  # pixels
    max_bbox_ratio: float = 0.9  # max bbox size relative to image


@dataclass
class TransformConfig:
    """Configuration for a single transform."""
    name: str
    probability: float = 0.5
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline."""
    enabled: bool = True
    augment_factor: int = 2  # versions per image
    num_workers: int = 1  # parallel workers
    transforms: list = field(default_factory=lambda: [
        TransformConfig("horizontal_flip", 0.5),
        TransformConfig("rotate", 0.3, {"limit": 15}),
        TransformConfig("brightness_contrast", 0.3, {"brightness": 0.2, "contrast": 0.2}),
    ])
