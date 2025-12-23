"""
SLAM Service - Simultaneous Localization and Mapping.
Provides spatial awareness, camera pose estimation, and 3D mapping integration.

Implements lightweight visual odometry using optical flow for egocentric devices.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DevicePose:
    """Represent device position and rotation relative to start."""
    timestamp: float
    # Cumulative position (normalized units)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0  # Depth (estimated from motion magnitude)
    # Rotation (degrees)
    yaw: float = 0.0    # Left/right rotation
    pitch: float = 0.0  # Up/down rotation
    roll: float = 0.0   # Tilt
    # Deltas from last frame
    delta_x: float = 0.0
    delta_y: float = 0.0
    rotation_deg: float = 0.0
    # Confidence in pose estimate (0-1)
    confidence: float = 1.0


@dataclass
class SpatialAnchor:
    """A persistent anchor for a detection in the local session."""
    id: int
    label: str
    # World-relative coordinates (accumulated from pose)
    world_coords: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Screen-relative coordinates at time of detection
    relative_coords: Tuple[float, float] = (0.5, 0.5)
    confidence: float = 0.8
    # Tracking info
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    frame_count: int = 1


class OpticalFlowTracker:
    """
    Lightweight optical flow tracker for pose estimation.
    Uses Lucas-Kanade sparse optical flow for efficiency on mobile devices.
    """
    
    def __init__(self, max_corners: int = 100, quality_level: float = 0.3):
        self.max_corners = max_corners
        self.quality_level = quality_level
        
        # Shi-Tomasi corner detection params
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=7,
            blockSize=7
        )
        
        # Lucas-Kanade optical flow params
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        self.prev_gray = None
        self.prev_points = None
    
    def compute_flow(self, frame: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Compute optical flow and estimate camera motion.
        
        Returns:
            (delta_x, delta_y, rotation_deg, confidence)
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # First frame - initialize
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )
            return 0.0, 0.0, 0.0, 1.0
        
        # Detect features if we lost too many
        if self.prev_points is None or len(self.prev_points) < 10:
            self.prev_points = cv2.goodFeaturesToTrack(
                self.prev_gray, mask=None, **self.feature_params
            )
            if self.prev_points is None:
                self.prev_gray = gray
                return 0.0, 0.0, 0.0, 0.0
        
        # Calculate optical flow
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )
        
        if next_points is None:
            self.prev_gray = gray
            self.prev_points = None
            return 0.0, 0.0, 0.0, 0.0
        
        # Select good points
        good_old = self.prev_points[status == 1]
        good_new = next_points[status == 1]
        
        if len(good_old) < 4:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )
            return 0.0, 0.0, 0.0, 0.0
        
        # Compute median motion (robust to outliers)
        motion = good_new - good_old
        median_motion = np.median(motion, axis=0)
        
        # Normalize by image size
        h, w = gray.shape[:2]
        delta_x = median_motion[0] / w
        delta_y = median_motion[1] / h
        
        # Estimate rotation from flow field divergence
        # Use the difference between left and right half motions
        mid_x = w // 2
        left_mask = good_old[:, 0] < mid_x
        right_mask = ~left_mask
        
        rotation_deg = 0.0
        if np.sum(left_mask) > 2 and np.sum(right_mask) > 2:
            left_motion = np.median(motion[left_mask, 0])
            right_motion = np.median(motion[right_mask, 0])
            # Opposite motions indicate rotation
            rotation_signal = right_motion - left_motion
            rotation_deg = rotation_signal * 0.1  # Scale factor
        
        # Confidence based on number of tracked points
        confidence = min(1.0, len(good_old) / self.max_corners)
        
        # Update for next frame
        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2)
        
        return delta_x, delta_y, rotation_deg, confidence
    
    def reset(self):
        """Reset tracker state."""
        self.prev_gray = None
        self.prev_points = None


class SlamService:
    """
    Lightweight Spatial Intelligence for Mobile & Wearables.
    
    Optimized for:
    1. Egocentric video (Glasses/Phones).
    2. Temporal Smoothing: Keeping YOLO boxes stable during head movement.
    3. Low-power tracking: Reducing inference by "locking" boxes to pixel features.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.imu_enabled = self.config.get("imu_enabled", False)
        
        # Optical flow tracker
        self.tracker = OpticalFlowTracker(
            max_corners=self.config.get("max_corners", 100),
            quality_level=self.config.get("quality_level", 0.3),
        )
        
        # Cumulative pose
        self._pose = DevicePose(timestamp=time.time())
        
        # Spatial anchors
        self._anchors: List[SpatialAnchor] = []
        self._next_anchor_id = 0
        
        # Smoothing factor for pose updates
        self._smoothing = self.config.get("smoothing", 0.7)
        
        logger.info("SLAM Service initialized (Optical Flow + IMU fusion ready)")

    @property
    def active_anchors(self) -> List[SpatialAnchor]:
        """Get active anchors (alias for compatibility)."""
        return self._anchors

    def update_pose(
        self, 
        frame: np.ndarray, 
        imu_data: Optional[Dict[str, float]] = None
    ) -> DevicePose:
        """
        Update local device pose using Optical Flow and optional IMU.
        
        Args:
            frame: BGR image as numpy array
            imu_data: Optional dict with accel_x/y/z and gyro_x/y/z
        
        Returns:
            Updated DevicePose with cumulative position and rotation
        """
        current_time = time.time()
        
        # Compute optical flow
        delta_x, delta_y, rotation_deg, confidence = self.tracker.compute_flow(frame)
        
        # Fuse with IMU if available
        if self.imu_enabled and imu_data:
            # Weight optical flow and IMU based on confidence
            gyro_yaw = imu_data.get("gyro_z", 0.0) * 0.01  # Scale gyro
            rotation_deg = self._smoothing * rotation_deg + (1 - self._smoothing) * gyro_yaw
            
            # Could also fuse accelerometer for drift correction
            # accel_x = imu_data.get("accel_x", 0.0)
        
        # Smooth deltas
        smoothed_dx = self._smoothing * delta_x
        smoothed_dy = self._smoothing * delta_y
        
        # Update cumulative pose
        self._pose = DevicePose(
            timestamp=current_time,
            x=self._pose.x + smoothed_dx,
            y=self._pose.y + smoothed_dy,
            z=self._pose.z,  # TODO: Estimate depth from motion parallax
            yaw=self._pose.yaw + rotation_deg,
            pitch=self._pose.pitch,
            roll=self._pose.roll,
            delta_x=smoothed_dx,
            delta_y=smoothed_dy,
            rotation_deg=rotation_deg,
            confidence=confidence,
        )
        
        return self._pose

    def anchor_detection(self, detection: Any, pose: DevicePose) -> SpatialAnchor:
        """
        'Lock' a detection to a spatial anchor so it remains persistent
        even if it leaves the FOV temporarily.
        
        Args:
            detection: Object with class_name, confidence, bbox attributes
            pose: Current device pose
        
        Returns:
            SpatialAnchor with world coordinates
        """
        # Extract detection center (normalized screen coords)
        bbox = getattr(detection, 'bbox', [0, 0, 0, 0])
        if len(bbox) >= 4:
            # xyxy format -> center
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            # Normalize if in pixel coords (assume 640x480 if > 1)
            if cx > 1 or cy > 1:
                cx = cx / 640  # Approximate normalization
                cy = cy / 480
        else:
            cx, cy = 0.5, 0.5
        
        # Transform screen coords to world coords using pose
        # Simple model: world = screen + pose offset
        world_x = cx - 0.5 + pose.x  # Center-relative
        world_y = cy - 0.5 + pose.y
        world_z = pose.z
        
        # Apply rotation (simplified 2D rotation)
        yaw_rad = np.radians(pose.yaw)
        rotated_x = world_x * np.cos(yaw_rad) - world_y * np.sin(yaw_rad)
        rotated_y = world_x * np.sin(yaw_rad) + world_y * np.cos(yaw_rad)
        
        anchor = SpatialAnchor(
            id=self._next_anchor_id,
            label=getattr(detection, 'class_name', 'unknown'),
            world_coords=(rotated_x, rotated_y, world_z),
            relative_coords=(cx, cy),
            confidence=getattr(detection, 'confidence', 0.8),
            first_seen=time.time(),
            last_seen=time.time(),
        )
        
        self._next_anchor_id += 1
        self._anchors.append(anchor)
        
        return anchor

    def find_anchor_by_bbox(
        self, 
        bbox: Tuple[float, float, float, float],
        threshold: float = 0.3
    ) -> Optional[SpatialAnchor]:
        """
        Find existing anchor that matches a bounding box (for re-identification).
        
        Args:
            bbox: (x1, y1, x2, y2) or (cx, cy, w, h)
            threshold: Maximum distance to consider a match
        
        Returns:
            Matching anchor or None
        """
        # Get bbox center
        if len(bbox) >= 4:
            cx = (bbox[0] + bbox[2]) / 2 if bbox[2] > bbox[0] else bbox[0]
            cy = (bbox[1] + bbox[3]) / 2 if bbox[3] > bbox[1] else bbox[1]
        else:
            return None
        
        # Transform to world coords
        world_x = cx - 0.5 + self._pose.x
        world_y = cy - 0.5 + self._pose.y
        
        # Find closest anchor
        best_anchor = None
        best_dist = threshold
        
        for anchor in self._anchors:
            ax, ay, _ = anchor.world_coords
            dist = np.sqrt((world_x - ax)**2 + (world_y - ay)**2)
            if dist < best_dist:
                best_dist = dist
                best_anchor = anchor
        
        return best_anchor

    def update_anchor(self, anchor_id: int, detection: Any) -> bool:
        """Update an existing anchor with new observation."""
        for anchor in self._anchors:
            if anchor.id == anchor_id:
                anchor.last_seen = time.time()
                anchor.frame_count += 1
                anchor.confidence = max(
                    anchor.confidence,
                    getattr(detection, 'confidence', 0.5)
                )
                return True
        return False

    def get_active_map(self, max_staleness: float = 30.0) -> List[SpatialAnchor]:
        """
        Return the current session-local map of detections.
        
        Args:
            max_staleness: Maximum seconds since last seen
        
        Returns:
            List of active (non-stale) anchors
        """
        current_time = time.time()
        return [
            a for a in self._anchors 
            if (current_time - a.last_seen) < max_staleness
        ]

    def cleanup_stale(self, max_staleness: float = 60.0) -> int:
        """Remove stale anchors. Returns count removed."""
        current_time = time.time()
        original_count = len(self._anchors)
        self._anchors = [
            a for a in self._anchors
            if (current_time - a.last_seen) < max_staleness
        ]
        return original_count - len(self._anchors)

    def reset(self):
        """Reset all state."""
        self.tracker.reset()
        self._pose = DevicePose(timestamp=time.time())
        self._anchors = []
        self._next_anchor_id = 0

    def get_pose(self) -> DevicePose:
        """Get current cumulative pose."""
        return self._pose
