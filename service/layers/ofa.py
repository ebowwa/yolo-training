"""
Once-for-All (OFA) Elastic Network Support.

Based on: "Once-for-All: Train One Network and Specialize it for Efficient
Deployment" (arXiv:1908.09791)

Key Concepts:
- Train a single overparameterized "supernet"
- Extract specialized sub-networks without retraining
- Progressive shrinking: depth → width → kernel size
- Hardware-aware sub-network selection

Benefits:
- O(10^19) possible sub-networks from one training
- No architecture search or retraining needed for deployment
- Supports diverse hardware constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import random


@dataclass
class SubNetworkConfig:
    """Configuration for a specific sub-network."""
    # Depth: number of layers to use per stage
    depth: List[int] = field(default_factory=lambda: [2, 3, 4, 3])
    # Width: channel multiplier per stage
    width: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    # Kernel size per layer (3, 5, or 7 typically)
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    # Resolution multiplier
    resolution: int = 224


@dataclass
class OFAConfig:
    """Configuration for OFA training."""
    # Maximum values (supernet)
    max_depth: List[int] = field(default_factory=lambda: [4, 4, 6, 4])
    max_width: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    supported_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    supported_resolutions: List[int] = field(default_factory=lambda: [128, 160, 192, 224])
    
    # Progressive shrinking schedule
    kernel_shrink_epochs: int = 25
    depth_shrink_epochs: int = 50
    width_shrink_epochs: int = 75


class ElasticKernel(nn.Module):
    """
    Elastic kernel that supports multiple kernel sizes.
    
    Uses kernel transformation to extract smaller kernels from larger ones.
    A 7x7 kernel contains a 5x5 kernel contains a 3x3 kernel at the center.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_kernel_size: int = 7,
        supported_sizes: List[int] = None,
        stride: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_kernel_size = max_kernel_size
        self.supported_sizes = supported_sizes or [3, 5, 7]
        self.stride = stride
        self.groups = groups
        
        # Full kernel (max size)
        self.kernel = nn.Parameter(
            torch.zeros(out_channels, in_channels // groups, max_kernel_size, max_kernel_size)
        )
        nn.init.kaiming_uniform_(self.kernel)
        
        # Kernel transformation matrices for extracting smaller kernels
        self.register_buffer("transform_3", self._get_transform_matrix(max_kernel_size, 3))
        self.register_buffer("transform_5", self._get_transform_matrix(max_kernel_size, 5))
        
        # Current active kernel size
        self._active_kernel_size = max_kernel_size
    
    def _get_transform_matrix(self, src_size: int, dst_size: int) -> torch.Tensor:
        """Get matrix to extract smaller kernel from larger kernel."""
        if src_size == dst_size:
            return torch.eye(src_size * src_size)
        
        # Simple center extraction
        offset = (src_size - dst_size) // 2
        transform = torch.zeros(dst_size * dst_size, src_size * src_size)
        
        for i in range(dst_size):
            for j in range(dst_size):
                src_i = i + offset
                src_j = j + offset
                dst_idx = i * dst_size + j
                src_idx = src_i * src_size + src_j
                transform[dst_idx, src_idx] = 1.0
        
        return transform
    
    def set_kernel_size(self, kernel_size: int):
        """Set the active kernel size."""
        if kernel_size not in self.supported_sizes:
            raise ValueError(f"Kernel size {kernel_size} not in {self.supported_sizes}")
        self._active_kernel_size = kernel_size
    
    def get_active_kernel(self) -> torch.Tensor:
        """Get the currently active kernel weights."""
        if self._active_kernel_size == self.max_kernel_size:
            return self.kernel
        
        # Extract smaller kernel
        batch_shape = self.kernel.shape[:2]
        flat_kernel = self.kernel.view(*batch_shape, -1)
        
        if self._active_kernel_size == 3:
            transform = self.transform_3
        elif self._active_kernel_size == 5:
            transform = self.transform_5
        else:
            return self.kernel
        
        # Apply transform
        small_kernel = torch.matmul(flat_kernel, transform.T)
        return small_kernel.view(*batch_shape, self._active_kernel_size, self._active_kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.get_active_kernel()
        padding = self._active_kernel_size // 2
        
        return F.conv2d(
            x, kernel,
            stride=self.stride,
            padding=padding,
            groups=self.groups,
        )


class ElasticWidth(nn.Module):
    """
    Elastic linear/conv layer that supports multiple width ratios.
    
    During training, randomly samples a width ratio.
    During inference, uses a fixed ratio.
    """
    
    def __init__(
        self,
        max_in_features: int,
        max_out_features: int,
        supported_ratios: List[float] = None,
        bias: bool = True,
    ):
        super().__init__()
        
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.supported_ratios = supported_ratios or [0.25, 0.5, 0.75, 1.0]
        
        self.weight = nn.Parameter(torch.zeros(max_out_features, max_in_features))
        nn.init.kaiming_uniform_(self.weight)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(max_out_features))
        else:
            self.register_parameter("bias", None)
        
        self._active_in = max_in_features
        self._active_out = max_out_features
    
    def set_width(self, in_ratio: float, out_ratio: float):
        """Set the active width ratios."""
        self._active_in = int(self.max_in_features * in_ratio)
        self._active_out = int(self.max_out_features * out_ratio)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight[:self._active_out, :self._active_in]
        bias = self.bias[:self._active_out] if self.bias is not None else None
        
        # Slice input if needed
        if x.size(-1) > self._active_in:
            x = x[..., :self._active_in]
        
        return F.linear(x, weight, bias)


class ElasticBlock(nn.Module):
    """
    A single elastic block that can be skipped (depth elasticity).
    
    When skipped, uses identity mapping or projection if dimensions differ.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 6.0,
    ):
        super().__init__()
        
        hidden_dim = int(in_channels * expand_ratio)
        
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Main path
        self.conv = nn.Sequential(
            # Expand
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Depthwise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Project
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # Skip projection (if needed)
        if not self.use_residual and in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        else:
            self.skip = None
        
        self._is_active = True
    
    def set_active(self, active: bool):
        """Enable or disable this block (depth elasticity)."""
        self._is_active = active
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_active:
            # Skip this block
            if self.skip is not None:
                return self.skip(x)
            return x
        
        out = self.conv(x)
        
        if self.use_residual:
            out = out + x
        
        return out


class OFASubNetworkExtractor:
    """
    Extracts and configures sub-networks from a trained OFA supernet.
    
    Usage:
        extractor = OFASubNetworkExtractor(supernet)
        
        # Sample random sub-network
        config = extractor.sample_random()
        extractor.set_config(config)
        
        # Run inference with sub-network
        output = supernet(input)
        
        # Or get standalone sub-network
        subnet = extractor.extract_standalone(config)
    """
    
    def __init__(self, supernet: nn.Module, ofa_config: OFAConfig):
        self.supernet = supernet
        self.config = ofa_config
        
        # Find elastic modules
        self.elastic_kernels = []
        self.elastic_widths = []
        self.elastic_blocks = []
        
        for name, module in supernet.named_modules():
            if isinstance(module, ElasticKernel):
                self.elastic_kernels.append((name, module))
            elif isinstance(module, ElasticWidth):
                self.elastic_widths.append((name, module))
            elif isinstance(module, ElasticBlock):
                self.elastic_blocks.append((name, module))
    
    def sample_random(self) -> SubNetworkConfig:
        """Sample a random sub-network configuration."""
        return SubNetworkConfig(
            depth=[random.randint(1, max_d) for max_d in self.config.max_depth],
            width=[random.choice([0.5, 0.75, 1.0]) for _ in self.config.max_width],
            kernel_sizes=[random.choice(self.config.supported_kernel_sizes) 
                         for _ in range(len(self.elastic_kernels))],
            resolution=random.choice(self.config.supported_resolutions),
        )
    
    def sample_smallest(self) -> SubNetworkConfig:
        """Get the smallest possible sub-network."""
        return SubNetworkConfig(
            depth=[1] * len(self.config.max_depth),
            width=[0.25] * len(self.config.max_width),
            kernel_sizes=[min(self.config.supported_kernel_sizes)] * len(self.elastic_kernels),
            resolution=min(self.config.supported_resolutions),
        )
    
    def sample_largest(self) -> SubNetworkConfig:
        """Get the largest possible sub-network (full supernet)."""
        return SubNetworkConfig(
            depth=self.config.max_depth.copy(),
            width=self.config.max_width.copy(),
            kernel_sizes=[max(self.config.supported_kernel_sizes)] * len(self.elastic_kernels),
            resolution=max(self.config.supported_resolutions),
        )
    
    def set_config(self, subnet_config: SubNetworkConfig):
        """Configure the supernet to operate as the specified sub-network."""
        # Set kernel sizes
        for i, (name, module) in enumerate(self.elastic_kernels):
            if i < len(subnet_config.kernel_sizes):
                module.set_kernel_size(subnet_config.kernel_sizes[i])
        
        # Set widths
        for i, (name, module) in enumerate(self.elastic_widths):
            if i < len(subnet_config.width):
                module.set_width(subnet_config.width[i], subnet_config.width[i])
        
        # Set depth (activate/deactivate blocks)
        block_idx = 0
        for stage_idx, stage_depth in enumerate(subnet_config.depth):
            max_depth = self.config.max_depth[stage_idx]
            for d in range(max_depth):
                if block_idx < len(self.elastic_blocks):
                    name, module = self.elastic_blocks[block_idx]
                    module.set_active(d < stage_depth)
                    block_idx += 1
    
    def get_complexity(self, subnet_config: SubNetworkConfig) -> Dict[str, float]:
        """Estimate FLOPs and parameters for a sub-network config."""
        # Simplified estimation
        base_flops = 1e9  # Base FLOPs for full network
        base_params = 1e7  # Base params for full network
        
        # Scale by depth
        depth_ratio = sum(subnet_config.depth) / sum(self.config.max_depth)
        
        # Scale by width (squared for FLOPs, linear for params)
        width_ratio = sum(subnet_config.width) / len(subnet_config.width)
        
        # Scale by resolution (squared)
        res_ratio = (subnet_config.resolution / 224) ** 2
        
        # Scale by kernel size
        avg_kernel = sum(subnet_config.kernel_sizes) / max(1, len(subnet_config.kernel_sizes))
        kernel_ratio = (avg_kernel / 5) ** 2
        
        flops = base_flops * depth_ratio * (width_ratio ** 2) * res_ratio * kernel_ratio
        params = base_params * depth_ratio * width_ratio
        
        return {
            "estimated_flops": flops,
            "estimated_params": params,
            "depth_ratio": depth_ratio,
            "width_ratio": width_ratio,
            "resolution": subnet_config.resolution,
        }


def create_progressive_shrinking_schedule(
    ofa_config: OFAConfig,
    total_epochs: int,
) -> List[Dict]:
    """
    Create a progressive shrinking training schedule.
    
    Order: Kernel → Depth → Width (as in OFA paper)
    
    Returns:
        List of dicts with epoch ranges and sampling constraints
    """
    schedule = []
    
    # Phase 1: Full network (warmup)
    schedule.append({
        "epoch_start": 0,
        "epoch_end": ofa_config.kernel_shrink_epochs,
        "sample_kernel": False,  # Use max kernel
        "sample_depth": False,   # Use max depth
        "sample_width": False,   # Use max width
    })
    
    # Phase 2: Kernel shrinking
    schedule.append({
        "epoch_start": ofa_config.kernel_shrink_epochs,
        "epoch_end": ofa_config.depth_shrink_epochs,
        "sample_kernel": True,
        "sample_depth": False,
        "sample_width": False,
    })
    
    # Phase 3: Depth shrinking
    schedule.append({
        "epoch_start": ofa_config.depth_shrink_epochs,
        "epoch_end": ofa_config.width_shrink_epochs,
        "sample_kernel": True,
        "sample_depth": True,
        "sample_width": False,
    })
    
    # Phase 4: Width shrinking (full OFA)
    schedule.append({
        "epoch_start": ofa_config.width_shrink_epochs,
        "epoch_end": total_epochs,
        "sample_kernel": True,
        "sample_depth": True,
        "sample_width": True,
    })
    
    return schedule
