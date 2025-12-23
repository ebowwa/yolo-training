"""
Neural Network Layers for Detection.

Provides building blocks for YOLO-style detection architectures:
- Backbone: ResNet feature extraction
- Neck: PAN-FPN feature fusion
- Head: Detection and density map heads
- Losses: CIoU, Focal, Distillation losses
- PEFT: LoRA, BitFit, Adapters for efficient fine-tuning
"""

from .common import Conv, Bottleneck, C2f, Concat, SPPF, DilatedConv, MultiDilatedBlock
from .backbone import ResNetBackbone
from .neck import PANFPN
from .head import Detect, DensityHead, DFL, dist2bbox, make_anchors
from .losses import (
    bbox_iou,
    CIoULoss,
    FocalLoss,
    BCEWithLogitsFocalLoss,
    DetectionLoss,
    DensityLoss,
    DistillationLoss,
    FeatureDistillationLoss,
    SimOTAAssigner,
)
from .peft import (
    LoRALinear,
    Adapter,
    apply_lora,
    apply_bitfit,
    freeze_except,
    count_parameters,
    get_trainable_parameters,
    GradientCheckpointWrapper,
)

__all__ = [
    # Common blocks
    "Conv", "Bottleneck", "C2f", "Concat", "SPPF", "DilatedConv", "MultiDilatedBlock",
    # Backbone
    "ResNetBackbone",
    # Neck
    "PANFPN",
    # Head
    "Detect", "DensityHead", "DFL", "dist2bbox", "make_anchors",
    # Losses
    "bbox_iou", "CIoULoss", "FocalLoss", "BCEWithLogitsFocalLoss",
    "DetectionLoss", "DensityLoss", "DistillationLoss", "FeatureDistillationLoss",
    "SimOTAAssigner",
    # PEFT
    "LoRALinear", "Adapter", "apply_lora", "apply_bitfit",
    "freeze_except", "count_parameters", "get_trainable_parameters",
    "GradientCheckpointWrapper",
]
