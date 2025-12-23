"""
Loss functions for object detection training.
Includes CIoU loss for bounding box regression and focal loss for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, 
             GIoU: bool = False, DIoU: bool = False, CIoU: bool = False, 
             eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate IoU between box1 and box2.
    
    Args:
        box1: (N, 4) tensor of boxes
        box2: (M, 4) tensor of boxes
        xywh: If True, boxes are (x, y, w, h), else (x1, y1, x2, y2)
        GIoU, DIoU, CIoU: Use respective IoU variants
        eps: Small value for numerical stability
    
    Returns:
        IoU tensor of shape (N, M) or (N,) if boxes aligned
    """
    # Convert xywh to xyxy
    if xywh:
        x1, y1, w1, h1 = box1.chunk(4, dim=-1)
        x2, y2, w2, h2 = box2.chunk(4, dim=-1)
        b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
        b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
        b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=-1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    if CIoU or DIoU or GIoU:
        # Convex (smallest enclosing box) width/height
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        
        if CIoU or DIoU:
            # Diagonal distance squared
            c2 = cw ** 2 + ch ** 2 + eps
            # Center distance squared
            rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 +
                    (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4
            if CIoU:
                # Aspect ratio consistency
                v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2  # DIoU
        # GIoU
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    
    return iou


class CIoULoss(nn.Module):
    """Complete IoU Loss for bounding box regression."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted boxes (N, 4) in xywh format
            target: Target boxes (N, 4) in xywh format
        """
        iou = bbox_iou(pred, target, xywh=True, CIoU=True)
        loss = 1.0 - iou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for classification.
    Addresses class imbalance by down-weighting easy examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, C) logits
            target: (N,) class indices
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class BCEWithLogitsFocalLoss(nn.Module):
    """
    BCE with logits + focal weighting for multi-label classification.
    Used in YOLO-style detection heads.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, C) logits
            target: (N, C) binary targets
        """
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SimOTAAssigner:
    """
    Simplified OTA (Optimal Transport Assignment) for target assignment.
    
    Based on YOLOX: https://arxiv.org/abs/2107.08430
    Assigns ground truth boxes to predictions based on a cost matrix.
    """
    
    def __init__(self, center_radius: float = 2.5, topk: int = 10):
        """
        Args:
            center_radius: Radius for center prior (in grid cells)
            topk: Number of top candidates to consider per GT
        """
        self.center_radius = center_radius
        self.topk = topk
    
    def assign(
        self,
        pred_boxes: torch.Tensor,      # (N, 4) predicted boxes in xywh
        pred_scores: torch.Tensor,     # (N, nc) predicted class scores
        gt_boxes: torch.Tensor,        # (M, 4) ground truth boxes in xywh
        gt_classes: torch.Tensor,      # (M,) ground truth class indices
        anchor_centers: torch.Tensor,  # (N, 2) anchor point centers
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform SimOTA assignment.
        
        Returns:
            matched_gt_idx: (K,) indices of matched GT for each positive prediction
            matched_pred_idx: (K,) indices of positive predictions
            fg_mask: (N,) boolean mask of foreground predictions
        """
        device = pred_boxes.device
        n_pred = pred_boxes.shape[0]
        n_gt = gt_boxes.shape[0]
        
        if n_gt == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.zeros(n_pred, dtype=torch.bool, device=device),
            )
        
        # Step 1: Filter by center prior - predictions near GT centers
        gt_centers = gt_boxes[:, :2]  # (M, 2)
        
        # Compute distances from each prediction to each GT center
        distances = torch.cdist(anchor_centers, gt_centers)  # (N, M)
        
        # Get candidates within radius
        is_in_center = distances < self.center_radius
        candidate_mask = is_in_center.any(dim=1)  # (N,)
        
        if not candidate_mask.any():
            # Fallback: use all predictions
            candidate_mask = torch.ones(n_pred, dtype=torch.bool, device=device)
        
        candidate_idx = candidate_mask.nonzero(as_tuple=True)[0]
        n_candidates = len(candidate_idx)
        
        # Step 2: Compute cost matrix for candidates
        cand_pred_boxes = pred_boxes[candidate_idx]  # (C, 4)
        cand_pred_scores = pred_scores[candidate_idx]  # (C, nc)
        
        # IoU cost
        iou = bbox_iou(
            cand_pred_boxes.unsqueeze(1),  # (C, 1, 4)
            gt_boxes.unsqueeze(0),          # (1, M, 4)
            xywh=True
        ).squeeze(-1)  # (C, M)
        iou_cost = -torch.log(iou + 1e-8)
        
        # Classification cost
        gt_onehot = F.one_hot(gt_classes, cand_pred_scores.shape[1]).float()  # (M, nc)
        cls_cost = F.binary_cross_entropy_with_logits(
            cand_pred_scores.unsqueeze(1).expand(-1, n_gt, -1),
            gt_onehot.unsqueeze(0).expand(n_candidates, -1, -1),
            reduction='none'
        ).sum(dim=-1)  # (C, M)
        
        # Total cost
        cost = iou_cost + cls_cost * 3.0
        
        # Step 3: Dynamic k selection based on IoU
        matching_matrix = torch.zeros(n_candidates, n_gt, dtype=torch.bool, device=device)
        
        for gt_idx in range(n_gt):
            gt_iou = iou[:, gt_idx]
            
            # Dynamic k: use IoU to determine how many to match
            dynamic_k = max(1, int(gt_iou.sum().item()))
            dynamic_k = min(dynamic_k, self.topk, n_candidates)
            
            # Get top-k lowest cost candidates for this GT
            _, topk_idx = cost[:, gt_idx].topk(dynamic_k, largest=False)
            matching_matrix[topk_idx, gt_idx] = True
        
        # Step 4: Resolve conflicts (one prediction -> one GT)
        # For predictions matching multiple GTs, keep the one with lowest cost
        matched_count = matching_matrix.sum(dim=1)
        conflicts = matched_count > 1
        
        if conflicts.any():
            conflict_idx = conflicts.nonzero(as_tuple=True)[0]
            for idx in conflict_idx:
                matched_gts = matching_matrix[idx].nonzero(as_tuple=True)[0]
                best_gt = matched_gts[cost[idx, matched_gts].argmin()]
                matching_matrix[idx] = False
                matching_matrix[idx, best_gt] = True
        
        # Step 5: Extract matches
        fg_mask_candidates = matching_matrix.any(dim=1)
        matched_gt_per_candidate = matching_matrix.float().argmax(dim=1)
        
        # Map back to original indices
        fg_mask = torch.zeros(n_pred, dtype=torch.bool, device=device)
        fg_mask[candidate_idx[fg_mask_candidates]] = True
        
        matched_pred_idx = candidate_idx[fg_mask_candidates]
        matched_gt_idx = matched_gt_per_candidate[fg_mask_candidates]
        
        return matched_gt_idx, matched_pred_idx, fg_mask


class DetectionLoss(nn.Module):
    """
    Combined loss for object detection with SimOTA target assignment.
    
    - Box loss: CIoU
    - Class loss: BCE with focal weighting
    - Objectness loss: BCE (positive for assigned predictions)
    """
    
    def __init__(self, nc: int = 80, box_weight: float = 7.5, 
                 cls_weight: float = 0.5, obj_weight: float = 1.0,
                 center_radius: float = 2.5, topk: int = 10):
        super().__init__()
        self.nc = nc
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        
        self.ciou_loss = CIoULoss(reduction='none')  # Per-sample loss
        self.cls_loss = BCEWithLogitsFocalLoss(reduction='none')
        self.obj_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        self.assigner = SimOTAAssigner(center_radius=center_radius, topk=topk)
    
    def forward(
        self, 
        pred: torch.Tensor,           # (B, N, 4+nc+1) or (B, N, 4+reg_max*4+nc)
        gt_boxes: torch.Tensor,       # (B, M, 4) ground truth boxes
        gt_classes: torch.Tensor,     # (B, M) ground truth classes
        anchor_centers: torch.Tensor, # (N, 2) anchor centers
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute detection loss with SimOTA assignment.
        
        Args:
            pred: Predictions (B, N, C) where C = 4 + nc + 1 (box, cls, obj)
            gt_boxes: Ground truth boxes (B, M, 4) in xywh format
            gt_classes: Ground truth class indices (B, M)
            anchor_centers: Anchor point centers (N, 2)
        
        Returns:
            total_loss, loss_dict
        """
        device = pred.device
        batch_size = pred.shape[0]
        n_pred = pred.shape[1]
        
        # Parse predictions
        pred_boxes = pred[..., :4]          # (B, N, 4)
        pred_cls = pred[..., 4:4+self.nc]   # (B, N, nc)
        pred_obj = pred[..., -1]            # (B, N)
        
        total_box_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        n_pos = 0
        
        for b in range(batch_size):
            # Get valid GTs for this image (filter padded zeros)
            valid_mask = gt_boxes[b].sum(dim=-1) > 0
            b_gt_boxes = gt_boxes[b][valid_mask]  # (m, 4)
            b_gt_classes = gt_classes[b][valid_mask]  # (m,)
            
            b_pred_boxes = pred_boxes[b]  # (N, 4)
            b_pred_cls = pred_cls[b]      # (N, nc)
            b_pred_obj = pred_obj[b]      # (N,)
            
            # Perform assignment
            matched_gt_idx, matched_pred_idx, fg_mask = self.assigner.assign(
                b_pred_boxes,
                b_pred_cls.sigmoid(),  # Use sigmoid scores for cost
                b_gt_boxes,
                b_gt_classes,
                anchor_centers,
            )
            
            n_matched = len(matched_pred_idx)
            n_pos += n_matched
            
            # Objectness loss (all predictions)
            obj_targets = fg_mask.float()
            total_obj_loss += self.obj_loss(b_pred_obj, obj_targets).mean()
            
            if n_matched == 0:
                continue
            
            # Box loss (matched predictions only)
            matched_pred_boxes = b_pred_boxes[matched_pred_idx]
            matched_gt_boxes = b_gt_boxes[matched_gt_idx]
            box_loss = self.ciou_loss(matched_pred_boxes, matched_gt_boxes)
            total_box_loss += box_loss.mean()
            
            # Classification loss (matched predictions only)
            matched_pred_cls = b_pred_cls[matched_pred_idx]
            cls_targets = F.one_hot(
                b_gt_classes[matched_gt_idx], self.nc
            ).float()
            cls_loss = F.binary_cross_entropy_with_logits(
                matched_pred_cls, cls_targets, reduction='none'
            ).sum(dim=-1)
            total_cls_loss += cls_loss.mean()
        
        # Normalize by batch and number of positives
        n_pos = max(1, n_pos)
        box_loss = total_box_loss / batch_size
        cls_loss = total_cls_loss / batch_size
        obj_loss = total_obj_loss / batch_size
        
        total = (self.box_weight * box_loss + 
                 self.cls_weight * cls_loss + 
                 self.obj_weight * obj_loss)
        
        return total, {
            'box_loss': box_loss.item(),
            'cls_loss': cls_loss.item(),
            'obj_loss': obj_loss.item(),
            'n_positives': n_pos,
            'total': total.item(),
        }


class DensityLoss(nn.Module):
    """
    Loss for density map counting.
    Uses MSE on density maps.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred_density: torch.Tensor, gt_density: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_density: Predicted density map (B, 1, H, W)
            gt_density: Ground truth density map (B, 1, H, W)
        """
        return self.mse(pred_density, gt_density)


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for training student models.
    Based on: https://arxiv.org/abs/2307.07483 (Multimodal Distillation)
    
    Combines:
    - Soft targets (KL divergence with temperature)
    - Hard targets (standard task loss)
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        """
        Args:
            temperature: Softmax temperature for soft targets (higher = softer)
            alpha: Weight for distillation loss (1-alpha for hard target loss)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                hard_targets: Optional[torch.Tensor] = None,
                hard_loss_fn: Optional[nn.Module] = None) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            student_logits: Student model output (B, C, ...)
            teacher_logits: Teacher model output (B, C, ...) - detached
            hard_targets: Optional ground truth labels
            hard_loss_fn: Optional loss function for hard targets
        
        Returns:
            total_loss, loss_dict
        """
        T = self.temperature
        
        # Flatten spatial dimensions if needed
        if student_logits.dim() > 2:
            student_flat = student_logits.view(student_logits.size(0), student_logits.size(1), -1)
            teacher_flat = teacher_logits.view(teacher_logits.size(0), teacher_logits.size(1), -1)
        else:
            student_flat = student_logits
            teacher_flat = teacher_logits
        
        # Soft targets: KL divergence with temperature
        soft_student = F.log_softmax(student_flat / T, dim=1)
        soft_teacher = F.softmax(teacher_flat.detach() / T, dim=1)
        
        distill_loss = F.kl_div(
            soft_student, soft_teacher, 
            reduction='batchmean'
        ) * (T ** 2)  # Scale by T^2 as per Hinton et al.
        
        # Optional hard targets
        if hard_targets is not None and hard_loss_fn is not None:
            hard_loss = hard_loss_fn(student_logits, hard_targets)
            total_loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss
            return total_loss, {
                'distill_loss': distill_loss.item(),
                'hard_loss': hard_loss.item(),
                'total': total_loss.item(),
            }
        
        return distill_loss, {
            'distill_loss': distill_loss.item(),
            'total': distill_loss.item(),
        }


class FeatureDistillationLoss(nn.Module):
    """
    Feature-level distillation for intermediate representations.
    Useful for transferring spatial knowledge from teacher to student.
    """
    
    def __init__(self, student_channels: List[int], teacher_channels: List[int]):
        """
        Args:
            student_channels: Channel dims for student features [P3, P4, P5]
            teacher_channels: Channel dims for teacher features [P3, P4, P5]
        """
        super().__init__()
        # Projection layers to align student features to teacher
        self.projections = nn.ModuleList([
            nn.Conv2d(s_ch, t_ch, 1) if s_ch != t_ch else nn.Identity()
            for s_ch, t_ch in zip(student_channels, teacher_channels)
        ])
    
    def forward(self, student_features: List[torch.Tensor], 
                teacher_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            student_features: List of student feature maps [P3, P4, P5]
            teacher_features: List of teacher feature maps [P3, P4, P5]
        """
        total_loss = 0.0
        for proj, s_feat, t_feat in zip(self.projections, student_features, teacher_features):
            # Project student to teacher dimension
            s_projected = proj(s_feat)
            # L2 loss on normalized features
            s_norm = F.normalize(s_projected, dim=1)
            t_norm = F.normalize(t_feat.detach(), dim=1)
            total_loss += F.mse_loss(s_norm, t_norm)
        
        return total_loss / len(self.projections)


# =============================================================================
# Interactive Counting (arxiv:2309.05277)
# =============================================================================

from dataclasses import dataclass

@dataclass
class RangeFeedback:
    """
    User feedback for a region of the density map.
    
    Based on: "Interactive Class-Agnostic Object Counting" (arxiv:2309.05277)
    
    The user selects a region and specifies a count range (e.g., "this area has 5-10 objects").
    """
    # Region bounds (normalized 0-1)
    x1: float
    y1: float
    x2: float
    y2: float
    # User-specified count range
    min_count: int
    max_count: int
    # Optional: confidence weight for this feedback
    weight: float = 1.0


class InteractiveCountingLoss(nn.Module):
    """
    Adaptation Loss for Interactive Class-Agnostic Object Counting.
    
    Based on: "Interactive Class-Agnostic Object Counting" (arxiv:2309.05277)
    
    This loss enables real-time adaptation of a counting model based on user feedback.
    The user provides feedback as count ranges for specific regions, and the loss
    penalizes predictions outside those ranges.
    
    Key insight: Uses a "soft boundary" penalty that:
    - Is 0 when predicted count is within [min_count, max_count]
    - Increases smoothly as the prediction moves outside the range
    
    Example:
        feedback = [
            RangeFeedback(x1=0.2, y1=0.3, x2=0.5, y2=0.6, min_count=5, max_count=10),
            RangeFeedback(x1=0.7, y1=0.7, x2=0.9, y2=0.9, min_count=0, max_count=2),
        ]
        loss = InteractiveCountingLoss()
        adaptation_loss = loss(density_map, feedback)
        adaptation_loss.backward()
    """
    
    def __init__(self, margin: float = 0.5, reduction: str = 'mean'):
        """
        Args:
            margin: Soft margin around the count range boundaries.
                    Larger values = smoother penalty gradient.
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self, 
        density_map: torch.Tensor,  # (B, 1, H, W) or (1, H, W) or (H, W)
        feedback: List[RangeFeedback],
    ) -> torch.Tensor:
        """
        Compute adaptation loss based on user feedback.
        
        Args:
            density_map: Predicted density map
            feedback: List of RangeFeedback from user
        
        Returns:
            Scalar loss tensor (differentiable)
        """
        if len(feedback) == 0:
            return torch.tensor(0.0, device=density_map.device, requires_grad=True)
        
        # Ensure 4D: (B, 1, H, W)
        if density_map.dim() == 2:
            density_map = density_map.unsqueeze(0).unsqueeze(0)
        elif density_map.dim() == 3:
            density_map = density_map.unsqueeze(0)
        
        B, C, H, W = density_map.shape
        losses = []
        
        for fb in feedback:
            # Convert normalized coords to pixel coords
            px1 = int(fb.x1 * W)
            py1 = int(fb.y1 * H)
            px2 = int(fb.x2 * W)
            py2 = int(fb.y2 * H)
            
            # Clamp to valid range
            px1, px2 = max(0, px1), min(W, px2)
            py1, py2 = max(0, py1), min(H, py2)
            
            if px2 <= px1 or py2 <= py1:
                continue
            
            # Extract region and compute count (sum of density)
            region = density_map[:, :, py1:py2, px1:px2]
            predicted_count = region.sum()
            
            # Soft boundary loss
            # Loss = 0 if min_count <= pred <= max_count
            # Loss = (pred - max_count)^2 if pred > max_count
            # Loss = (min_count - pred)^2 if pred < min_count
            min_c = float(fb.min_count) - self.margin
            max_c = float(fb.max_count) + self.margin
            
            if predicted_count < min_c:
                region_loss = (min_c - predicted_count) ** 2
            elif predicted_count > max_c:
                region_loss = (predicted_count - max_c) ** 2
            else:
                region_loss = torch.tensor(0.0, device=density_map.device)
            
            losses.append(region_loss * fb.weight)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=density_map.device, requires_grad=True)
        
        stacked = torch.stack(losses)
        
        if self.reduction == 'mean':
            return stacked.mean()
        elif self.reduction == 'sum':
            return stacked.sum()
        else:
            return stacked


class CountingRefinementAdapter(nn.Module):
    """
    Refinement Adapter for Interactive Counting.
    
    Based on: "Interactive Class-Agnostic Object Counting" (arxiv:2309.05277)
    
    A lightweight adapter module that can be attached to any density-based counter.
    Only the parameters in this adapter are updated during interactive adaptation,
    leaving the base model frozen.
    
    Architecture:
        Input density map -> Conv layers -> Residual refinement -> Output density map
    
    Example:
        base_model = DensityHead(...)  # Frozen during adaptation
        adapter = CountingRefinementAdapter(in_channels=1)
        
        # Inference with refinement
        density = base_model(features)
        refined_density = adapter(density)
        
        # Only adapter params are updated
        optimizer = torch.optim.Adam(adapter.parameters(), lr=0.001)
    """
    
    def __init__(
        self, 
        in_channels: int = 1, 
        hidden_channels: int = 32,
        num_layers: int = 3,
        residual: bool = True,
    ):
        """
        Args:
            in_channels: Input density map channels (usually 1)
            hidden_channels: Hidden layer channels
            num_layers: Number of refinement conv layers
            residual: If True, output = input + refinement (residual learning)
        """
        super().__init__()
        self.residual = residual
        
        layers = []
        
        # First layer: expand channels
        layers.append(nn.Conv2d(in_channels, hidden_channels, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        
        # Final layer: compress back to density
        layers.append(nn.Conv2d(hidden_channels, in_channels, 3, padding=1))
        
        self.refine = nn.Sequential(*layers)
        
        # Initialize to near-identity (small refinements initially)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for small initial refinement."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, density: torch.Tensor) -> torch.Tensor:
        """
        Refine a density map.
        
        Args:
            density: Input density map (B, 1, H, W)
        
        Returns:
            Refined density map (B, 1, H, W)
        """
        refinement = self.refine(density)
        
        if self.residual:
            # Residual connection: output = input + small refinement
            return density + refinement
        else:
            return refinement
    
    def adapt(
        self,
        density: torch.Tensor,
        feedback: List[RangeFeedback],
        lr: float = 0.001,
        steps: int = 10,
    ) -> torch.Tensor:
        """
        Perform quick adaptation based on user feedback.
        
        This is a convenience method that runs a few gradient steps
        to adapt the refinement module to the user's feedback.
        
        Args:
            density: Input density map from base model (detached)
            feedback: User feedback as list of RangeFeedback
            lr: Learning rate for adaptation
            steps: Number of gradient steps
        
        Returns:
            Refined density map after adaptation
        """
        loss_fn = InteractiveCountingLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        density = density.detach()  # Don't backprop into base model
        
        for _ in range(steps):
            optimizer.zero_grad()
            refined = self(density)
            loss = loss_fn(refined, feedback)
            if loss.item() > 0:
                loss.backward()
                optimizer.step()
            else:
                break  # Perfect fit, stop early
        
        return self(density)

