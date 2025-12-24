"""
Knowledge Distillation Module.

Based on: "Patient Knowledge Distillation for BERT Model Compression"
(arXiv:2012.06785) and general distillation literature.

Key Techniques:
- Soft label distillation with temperature scaling
- Patient (multi-layer) feature matching
- PKD-Last: distill from last K teacher layers
- PKD-Skip: distill from every K-th teacher layer
- Response-based, feature-based, and relation-based distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class DistillationStrategy(Enum):
    """Distillation strategy types."""
    RESPONSE = "response"      # Output logits only
    FEATURE = "feature"        # Intermediate features
    PATIENT = "patient"        # Multi-layer patient distillation
    ATTENTION = "attention"    # Attention map distillation


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    # Temperature for soft labels (higher = softer)
    temperature: float = 4.0
    # Weight balance between distillation and task loss
    alpha: float = 0.5  # alpha * distill_loss + (1-alpha) * task_loss
    # Distillation strategy
    strategy: DistillationStrategy = DistillationStrategy.PATIENT
    # For PKD: number of teacher layers to use
    num_teacher_layers: int = 6
    # PKD variant: "last" (last K layers) or "skip" (every K-th layer)
    pkd_variant: str = "skip"
    # Loss type for feature matching
    feature_loss: str = "mse"  # "mse" or "cosine"
    # Whether to normalize features before matching
    normalize_features: bool = True


class SoftLabelLoss(nn.Module):
    """
    Soft label distillation loss (Hinton et al., 2015).
    
    Uses temperature-scaled softmax to create softer probability
    distributions that reveal more about the learned relationships.
    """
    
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute soft label distillation loss.
        
        Args:
            student_logits: Raw logits from student (B, C)
            teacher_logits: Raw logits from teacher (B, C)
            
        Returns:
            Scalar loss value
        """
        # Temperature-scaled softmax
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence
        loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        
        # Scale by T^2 (as in original paper)
        return loss * (self.temperature ** 2)


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss for intermediate layer distillation.
    
    Matches the hidden states of student and teacher at specific layers.
    """
    
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        loss_type: str = "mse",
        normalize: bool = True,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
        
        # Projection if dimensions differ
        if student_dim != teacher_dim:
            self.projection = nn.Linear(student_dim, teacher_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute feature matching loss.
        
        Args:
            student_features: Features from student (B, *, D_s)
            teacher_features: Features from teacher (B, *, D_t)
            
        Returns:
            Scalar loss value
        """
        # Project student to teacher dimension
        student_proj = self.projection(student_features)
        
        # Normalize if requested
        if self.normalize:
            student_proj = F.normalize(student_proj, dim=-1)
            teacher_features = F.normalize(teacher_features, dim=-1)
        
        # Compute loss
        if self.loss_type == "mse":
            loss = F.mse_loss(student_proj, teacher_features)
        elif self.loss_type == "cosine":
            loss = 1 - F.cosine_similarity(student_proj, teacher_features, dim=-1).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class PatientDistillationLoss(nn.Module):
    """
    Patient Knowledge Distillation (PKD) loss.
    
    Distills knowledge from multiple teacher layers, not just the final output.
    This helps the student learn intermediate representations that lead to
    better generalization.
    
    Variants:
    - PKD-Last: Student layer i matches teacher layer (T - S + i)
      where T = num teacher layers, S = num student layers
    - PKD-Skip: Student layer i matches teacher layer (i * T // S)
    
    Example:
        Teacher: 12 layers, Student: 6 layers
        PKD-Last: Student [1,2,3,4,5,6] matches Teacher [7,8,9,10,11,12]
        PKD-Skip: Student [1,2,3,4,5,6] matches Teacher [2,4,6,8,10,12]
    """
    
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        num_student_layers: int,
        num_teacher_layers: int,
        variant: str = "skip",
        normalize: bool = True,
    ):
        super().__init__()
        self.num_student_layers = num_student_layers
        self.num_teacher_layers = num_teacher_layers
        self.variant = variant
        
        # Build layer mapping
        self.layer_mapping = self._build_layer_mapping()
        
        # Feature matchers for each student layer
        self.matchers = nn.ModuleList([
            FeatureMatchingLoss(student_dim, teacher_dim, normalize=normalize)
            for _ in range(num_student_layers)
        ])
    
    def _build_layer_mapping(self) -> Dict[int, int]:
        """Build mapping from student layer index to teacher layer index."""
        mapping = {}
        S = self.num_student_layers
        T = self.num_teacher_layers
        
        for i in range(S):
            if self.variant == "last":
                # Student layer i matches teacher layer (T - S + i)
                mapping[i] = T - S + i
            elif self.variant == "skip":
                # Student layer i matches teacher layer ((i + 1) * T // S - 1)
                mapping[i] = (i + 1) * T // S - 1
            else:
                raise ValueError(f"Unknown PKD variant: {self.variant}")
        
        return mapping
    
    def forward(
        self,
        student_hidden_states: List[torch.Tensor],
        teacher_hidden_states: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute patient distillation loss.
        
        Args:
            student_hidden_states: List of hidden states from student layers
            teacher_hidden_states: List of hidden states from teacher layers
            
        Returns:
            Scalar loss value (sum over matching layers)
        """
        total_loss = 0.0
        
        for student_idx, matcher in enumerate(self.matchers):
            teacher_idx = self.layer_mapping[student_idx]
            
            if student_idx < len(student_hidden_states) and teacher_idx < len(teacher_hidden_states):
                student_feat = student_hidden_states[student_idx]
                teacher_feat = teacher_hidden_states[teacher_idx].detach()
                
                total_loss = total_loss + matcher(student_feat, teacher_feat)
        
        return total_loss / max(1, len(self.matchers))


class AttentionDistillationLoss(nn.Module):
    """
    Attention map distillation loss.
    
    Transfers the attention patterns from teacher to student,
    helping the student learn what to focus on.
    """
    
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
    
    def forward(
        self,
        student_attention: torch.Tensor,  # (B, H_s, S, S)
        teacher_attention: torch.Tensor,  # (B, H_t, S, S)
    ) -> torch.Tensor:
        """
        Compute attention distillation loss.
        
        Args:
            student_attention: Attention weights from student
            teacher_attention: Attention weights from teacher
            
        Returns:
            Scalar loss value
        """
        # Average across heads if different numbers
        if student_attention.size(1) != teacher_attention.size(1):
            teacher_attention = teacher_attention.mean(dim=1, keepdim=True)
            teacher_attention = teacher_attention.expand_as(student_attention)
        
        # MSE between attention patterns
        loss = F.mse_loss(student_attention, teacher_attention.detach())
        
        return loss


class DistillationLoss(nn.Module):
    """
    Combined distillation loss supporting multiple strategies.
    
    Usage:
        loss_fn = DistillationLoss(config)
        
        # Forward pass through both models
        student_out = student(x)
        with torch.no_grad():
            teacher_out = teacher(x)
        
        # Compute distillation loss
        distill_loss = loss_fn(student_out, teacher_out)
        
        # Combine with task loss
        total_loss = config.alpha * distill_loss + (1 - config.alpha) * task_loss
    """
    
    def __init__(
        self,
        config: DistillationConfig,
        student_dim: Optional[int] = None,
        teacher_dim: Optional[int] = None,
        num_student_layers: Optional[int] = None,
        num_teacher_layers: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        
        # Soft label loss (always used)
        self.soft_label_loss = SoftLabelLoss(config.temperature)
        
        # Feature matching loss (for feature/patient strategies)
        if config.strategy in [DistillationStrategy.FEATURE, DistillationStrategy.PATIENT]:
            if student_dim is None or teacher_dim is None:
                raise ValueError("Need student_dim and teacher_dim for feature distillation")
            
            if config.strategy == DistillationStrategy.PATIENT:
                if num_student_layers is None or num_teacher_layers is None:
                    raise ValueError("Need layer counts for patient distillation")
                
                self.patient_loss = PatientDistillationLoss(
                    student_dim=student_dim,
                    teacher_dim=teacher_dim,
                    num_student_layers=num_student_layers,
                    num_teacher_layers=num_teacher_layers,
                    variant=config.pkd_variant,
                    normalize=config.normalize_features,
                )
            else:
                self.feature_loss = FeatureMatchingLoss(
                    student_dim=student_dim,
                    teacher_dim=teacher_dim,
                    loss_type=config.feature_loss,
                    normalize=config.normalize_features,
                )
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute combined distillation loss.
        
        Args:
            student_outputs: Dict with 'logits', optional 'hidden_states', 'attentions'
            teacher_outputs: Dict with same keys from teacher
            
        Returns:
            Scalar distillation loss
        """
        losses = {}
        
        # Soft label loss (response-based)
        if "logits" in student_outputs and "logits" in teacher_outputs:
            losses["soft_label"] = self.soft_label_loss(
                student_outputs["logits"],
                teacher_outputs["logits"],
            )
        
        # Feature loss
        if self.config.strategy == DistillationStrategy.PATIENT:
            if "hidden_states" in student_outputs and "hidden_states" in teacher_outputs:
                losses["patient"] = self.patient_loss(
                    student_outputs["hidden_states"],
                    teacher_outputs["hidden_states"],
                )
        elif self.config.strategy == DistillationStrategy.FEATURE:
            if "features" in student_outputs and "features" in teacher_outputs:
                losses["feature"] = self.feature_loss(
                    student_outputs["features"],
                    teacher_outputs["features"],
                )
        
        # Combine losses (equal weighting)
        total_loss = sum(losses.values()) / max(1, len(losses))
        
        return total_loss


def create_distillation_trainer(
    teacher: nn.Module,
    student: nn.Module,
    config: DistillationConfig,
) -> Dict:
    """
    Create a distillation training setup.
    
    Returns:
        Dict with loss_fn, and training utilities
    """
    # Freeze teacher
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Determine dimensions (heuristic)
    student_dim = None
    teacher_dim = None
    
    for name, module in student.named_modules():
        if isinstance(module, nn.Linear):
            student_dim = module.in_features
            break
    
    for name, module in teacher.named_modules():
        if isinstance(module, nn.Linear):
            teacher_dim = module.in_features
            break
    
    # Create loss function
    loss_fn = DistillationLoss(
        config=config,
        student_dim=student_dim,
        teacher_dim=teacher_dim,
        num_student_layers=6,  # Default, should be configured
        num_teacher_layers=config.num_teacher_layers,
    )
    
    return {
        "teacher": teacher,
        "student": student,
        "loss_fn": loss_fn,
        "config": config,
    }
