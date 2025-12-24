"""
MobileNet-style Depthwise Separable Convolutions.

Based on: "MobileNets: Efficient Convolutional Neural Networks for Mobile
Vision Applications" (arXiv:1704.04861)

Key Concepts:
- Depthwise convolution: Apply a single filter per input channel
- Pointwise convolution: 1x1 conv to combine channel information
- Width multiplier (α): Thin the network uniformly at each layer
- Resolution multiplier (ρ): Reduce input resolution

Computational savings:
- Standard conv: D_k² * M * N * D_f²
- Depthwise separable: D_k² * M * D_f² + M * N * D_f²
- Ratio: 1/N + 1/D_k² (typically 8-9x fewer operations)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class DepthwiseConv2d(nn.Module):
    """
    Depthwise convolution: each input channel is convolved separately.
    
    Also known as "grouped convolution" where groups = in_channels.
    """
    
    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,  # Same as input
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # Key: each channel processed separately
            bias=bias,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PointwiseConv2d(nn.Module):
    """
    Pointwise convolution: 1x1 conv to mix channel information.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution block from MobileNet.
    
    Splits a standard convolution into:
    1. Depthwise conv: spatial filtering within each channel
    2. Pointwise conv: channel mixing with 1x1 conv
    
    This provides ~8-9x computational savings vs standard conv.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of depthwise kernel
        stride: Stride for depthwise conv
        padding: Padding for depthwise conv
        bias: Whether to use bias
        use_bn: Whether to use batch normalization
        activation: Activation function ('relu', 'relu6', 'silu', None)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        bias: bool = False,
        use_bn: bool = True,
        activation: Optional[str] = "relu6",
    ):
        super().__init__()
        
        # Depthwise
        self.depthwise = DepthwiseConv2d(
            in_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        
        # Pointwise
        self.pointwise = PointwiseConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "relu6":
            self.activation = nn.ReLU6(inplace=True)
        elif activation == "silu" or activation == "swish":
            self.activation = nn.SiLU(inplace=True)
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # Pointwise
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        return x


class InvertedResidual(nn.Module):
    """
    Inverted Residual block from MobileNetV2.
    
    Structure:
    1. Pointwise expansion (expand channels)
    2. Depthwise filtering (spatial conv)
    3. Pointwise projection (reduce channels)
    
    Uses residual connection when stride=1 and in_channels=out_channels.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        stride: Stride for depthwise conv
        expand_ratio: Expansion ratio for hidden layer
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 6.0,
    ):
        super().__init__()
        
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Expansion (if expand_ratio > 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(
                hidden_dim, hidden_dim, 3,
                stride=stride, padding=1, groups=hidden_dim, bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])
        
        # Projection (linear, no activation)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


def apply_width_multiplier(channels: int, multiplier: float, divisor: int = 8) -> int:
    """
    Apply width multiplier and round to nearest divisor.
    
    Args:
        channels: Original number of channels
        multiplier: Width multiplier (α in MobileNet paper)
        divisor: Round to this divisor (for hardware efficiency)
        
    Returns:
        Adjusted number of channels
    """
    new_channels = int(channels * multiplier)
    # Round to nearest divisor
    new_channels = max(divisor, (new_channels + divisor // 2) // divisor * divisor)
    # Ensure we don't go below 0.9x of target
    if new_channels < 0.9 * channels * multiplier:
        new_channels += divisor
    return new_channels


def replace_conv_with_depthwise_separable(
    module: nn.Module,
    min_channels: int = 32,
) -> int:
    """
    Replace standard Conv2d layers with DepthwiseSeparableConv2d.
    
    Only replaces layers with kernel_size > 1 and sufficient channels.
    
    Args:
        module: Module to modify in-place
        min_channels: Minimum channels to consider for replacement
        
    Returns:
        Number of layers replaced
    """
    count = 0
    
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            kernel_size = child.kernel_size
            if isinstance(kernel_size, tuple):
                kernel_size = kernel_size[0]
            
            # Only replace if kernel > 1x1 and enough channels
            if kernel_size > 1 and child.in_channels >= min_channels:
                # Calculate padding to maintain spatial dims
                padding = kernel_size // 2
                
                new_conv = DepthwiseSeparableConv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=kernel_size,
                    stride=child.stride[0] if isinstance(child.stride, tuple) else child.stride,
                    padding=padding,
                    bias=child.bias is not None,
                )
                
                setattr(module, name, new_conv)
                count += 1
        else:
            count += replace_conv_with_depthwise_separable(child, min_channels)
    
    return count


def compute_flops_reduction(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    spatial_size: int,
) -> dict:
    """
    Compute FLOPs reduction from using depthwise separable conv.
    
    Returns:
        Dict with standard FLOPs, depthwise separable FLOPs, and ratio
    """
    # Standard conv FLOPs
    standard = kernel_size ** 2 * in_channels * out_channels * spatial_size ** 2
    
    # Depthwise separable FLOPs
    depthwise = kernel_size ** 2 * in_channels * spatial_size ** 2
    pointwise = in_channels * out_channels * spatial_size ** 2
    separable = depthwise + pointwise
    
    return {
        "standard_flops": standard,
        "depthwise_separable_flops": separable,
        "reduction_ratio": standard / separable,
        "savings_percent": (1 - separable / standard) * 100,
    }
