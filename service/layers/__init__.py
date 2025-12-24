"""
Neural Network Layers for Detection.

Provides building blocks for YOLO-style detection architectures:
- Backbone: ResNet feature extraction
- Neck: PAN-FPN feature fusion
- Head: Detection and density map heads
- Losses: CIoU, Focal, Distillation losses
- PEFT: LoRA, BitFit, Adapters for efficient fine-tuning
- Edge Optimization: Pruning, depthwise separable, OFA, distillation
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
    InteractiveCountingLoss,
    CountingRefinementAdapter,
    RangeFeedback,
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

# Edge Optimization Modules (arXiv papers integration)
from .pruning import (
    PruningConfig,
    HeadImportanceScorer,
    PrunedMultiHeadAttention,
    AttentionHeadPruner,
    apply_head_pruning,
)
from .depthwise_separable import (
    DepthwiseConv2d,
    PointwiseConv2d,
    DepthwiseSeparableConv2d,
    InvertedResidual,
    apply_width_multiplier,
    replace_conv_with_depthwise_separable,
    compute_flops_reduction,
)
from .ofa import (
    SubNetworkConfig,
    OFAConfig,
    ElasticKernel,
    ElasticWidth,
    ElasticBlock,
    OFASubNetworkExtractor,
    create_progressive_shrinking_schedule,
)
from .distillation import (
    DistillationStrategy,
    DistillationConfig,
    SoftLabelLoss,
    FeatureMatchingLoss,
    PatientDistillationLoss,
    AttentionDistillationLoss,
    DistillationLoss as PatientDistillationModule,
    create_distillation_trainer,
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
    # Interactive Counting (arxiv:2309.05277)
    "InteractiveCountingLoss", "CountingRefinementAdapter", "RangeFeedback",
    # PEFT
    "LoRALinear", "Adapter", "apply_lora", "apply_bitfit",
    "freeze_except", "count_parameters", "get_trainable_parameters",
    "GradientCheckpointWrapper",
    # Attention Head Pruning (arXiv:2201.08071)
    "PruningConfig", "HeadImportanceScorer", "PrunedMultiHeadAttention",
    "AttentionHeadPruner", "apply_head_pruning",
    # MobileNet Depthwise Separable (arXiv:1704.04861)
    "DepthwiseConv2d", "PointwiseConv2d", "DepthwiseSeparableConv2d",
    "InvertedResidual", "apply_width_multiplier",
    "replace_conv_with_depthwise_separable", "compute_flops_reduction",
    # Once-for-All Networks (arXiv:1908.09791)
    "SubNetworkConfig", "OFAConfig", "ElasticKernel", "ElasticWidth",
    "ElasticBlock", "OFASubNetworkExtractor", "create_progressive_shrinking_schedule",
    # Patient Knowledge Distillation (arXiv:2012.06785)
    "DistillationStrategy", "DistillationConfig", "SoftLabelLoss",
    "FeatureMatchingLoss", "PatientDistillationLoss", "AttentionDistillationLoss",
    "PatientDistillationModule", "create_distillation_trainer",
]
