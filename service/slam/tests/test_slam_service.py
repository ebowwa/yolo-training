"""
Tests for SLAM Service module.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Add paths for imports
_test_dir = Path(__file__).parent
_slam_dir = _test_dir.parent
_service_dir = _slam_dir.parent
sys.path.insert(0, str(_slam_dir))
sys.path.insert(0, str(_service_dir))

from slam_service import SlamService, DevicePose, SpatialAnchor


class TestDevicePose(unittest.TestCase):
    """Tests for DevicePose dataclass."""
    
    def test_default_values(self):
        """Test default pose values."""
        pose = DevicePose(timestamp=1.0)
        self.assertEqual(pose.timestamp, 1.0)
        self.assertEqual(pose.delta_x, 0.0)
        self.assertEqual(pose.delta_y, 0.0)
        self.assertEqual(pose.rotation_deg, 0.0)
    
    def test_custom_values(self):
        """Test pose with custom values."""
        pose = DevicePose(
            timestamp=2.5,
            delta_x=0.1,
            delta_y=-0.2,
            rotation_deg=15.0,
        )
        self.assertEqual(pose.timestamp, 2.5)
        self.assertEqual(pose.delta_x, 0.1)
        self.assertEqual(pose.delta_y, -0.2)
        self.assertEqual(pose.rotation_deg, 15.0)


class TestSpatialAnchor(unittest.TestCase):
    """Tests for SpatialAnchor dataclass."""
    
    def test_creation(self):
        """Test anchor creation."""
        anchor = SpatialAnchor(
            id=1,
            label="pothole",
            relative_coords=(0.5, 0.6),
            confidence=0.85,
        )
        self.assertEqual(anchor.id, 1)
        self.assertEqual(anchor.label, "pothole")
        self.assertEqual(anchor.relative_coords, (0.5, 0.6))
        self.assertEqual(anchor.confidence, 0.85)


class TestSlamService(unittest.TestCase):
    """Tests for SlamService class."""
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        svc = SlamService()
        self.assertFalse(svc.imu_enabled)
        self.assertEqual(svc.active_anchors, [])
    
    def test_init_with_imu_enabled(self):
        """Test initialization with IMU enabled."""
        svc = SlamService({"imu_enabled": True})
        self.assertTrue(svc.imu_enabled)
    
    def test_update_pose_returns_device_pose(self):
        """Test that update_pose returns a DevicePose."""
        svc = SlamService()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        pose = svc.update_pose(frame)
        
        self.assertIsInstance(pose, DevicePose)
        self.assertEqual(pose.timestamp, 0.0)  # Stub returns 0.0
    
    def test_update_pose_with_imu_data(self):
        """Test update_pose with IMU data."""
        svc = SlamService({"imu_enabled": True})
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        imu_data = {
            "accel_x": 0.1,
            "accel_y": 0.2,
            "accel_z": 9.8,
            "gyro_x": 0.01,
            "gyro_y": 0.02,
            "gyro_z": 0.0,
        }
        
        pose = svc.update_pose(frame, imu_data)
        
        self.assertIsInstance(pose, DevicePose)
    
    def test_anchor_detection(self):
        """Test anchoring a detection."""
        svc = SlamService()
        
        class MockDetection:
            class_name = "pothole"
            confidence = 0.9
            bbox = [100, 200, 150, 250]
        
        detection = MockDetection()
        pose = DevicePose(timestamp=1.0)
        
        anchor = svc.anchor_detection(detection, pose)
        
        self.assertIsInstance(anchor, SpatialAnchor)
        self.assertEqual(anchor.label, "pothole")
        self.assertEqual(len(svc.active_anchors), 1)
    
    def test_get_active_map(self):
        """Test retrieving the active map."""
        svc = SlamService()
        
        class MockDetection:
            class_name = "crack"
            confidence = 0.7
        
        # Add some anchors
        pose = DevicePose(timestamp=0.0)
        svc.anchor_detection(MockDetection(), pose)
        svc.anchor_detection(MockDetection(), pose)
        
        map_anchors = svc.get_active_map()
        
        self.assertEqual(len(map_anchors), 2)
        for a in map_anchors:
            self.assertIsInstance(a, SpatialAnchor)
    
    def test_multiple_anchors_accumulate(self):
        """Test that anchors accumulate in the map."""
        svc = SlamService()
        pose = DevicePose(timestamp=0.0)
        
        for i in range(5):
            class Det:
                class_name = f"obj_{i}"
            svc.anchor_detection(Det(), pose)
        
        self.assertEqual(len(svc.active_anchors), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
