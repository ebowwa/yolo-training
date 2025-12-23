"""
SLAM Module - Simultaneous Localization and Mapping.
Provides spatial awareness, camera pose estimation, and 3D mapping integration.

Optimized for egocentric video from wearables (glasses, phones).
"""

from .slam_service import SlamService, DevicePose, SpatialAnchor

__all__ = ["SlamService", "DevicePose", "SpatialAnchor"]
