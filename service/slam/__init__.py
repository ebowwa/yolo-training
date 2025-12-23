"""
SLAM Module - Simultaneous Localization and Mapping.
Provides spatial awareness, camera pose estimation, and 3D mapping integration.

Optimized for egocentric video from wearables (glasses, phones).
Uses Lucas-Kanade optical flow for lightweight pose estimation.
"""

from .slam_service import SlamService, DevicePose, SpatialAnchor, OpticalFlowTracker
from .object_cache import ObjectCache, CachedObject, get_object_cache

__all__ = [
    "SlamService",
    "DevicePose",
    "SpatialAnchor",
    "OpticalFlowTracker",
    "ObjectCache",
    "CachedObject",
    "get_object_cache",
]

