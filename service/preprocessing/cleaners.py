"""
Composable Cleaners - Individual cleaning operations.

Each cleaner is a single-responsibility class that can be composed
into a pipeline. All cleaners inherit from BaseCleaner.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class CleanResult:
    """Result from a cleaning operation on a single file."""
    image_path: str
    label_path: Optional[str]
    removed: bool = False
    fixed: bool = False
    error: Optional[str] = None


class BaseCleaner(ABC):
    """Base class for all cleaners."""

    @abstractmethod
    def clean(self, image_path: Path, label_path: Optional[Path]) -> CleanResult:
        """
        Clean a single image/label pair.

        Args:
            image_path: Path to image file.
            label_path: Path to label file (may not exist).

        Returns:
            CleanResult with status.
        """
        pass

    def clean_batch(self, pairs: List[Tuple[Path, Optional[Path]]]) -> List[CleanResult]:
        """Clean a batch of image/label pairs."""
        return [self.clean(img, lbl) for img, lbl in pairs]


class CorruptedImageCleaner(BaseCleaner):
    """Remove corrupted/unreadable images."""

    def clean(self, image_path: Path, label_path: Optional[Path]) -> CleanResult:
        result = CleanResult(str(image_path), str(label_path) if label_path else None)

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logging.warning(f"Removing corrupted image: {image_path}")
                os.remove(image_path)
                if label_path and label_path.exists():
                    os.remove(label_path)
                result.removed = True
        except Exception as e:
            logging.warning(f"Error reading {image_path}: {e}")
            os.remove(image_path)
            if label_path and label_path.exists():
                os.remove(label_path)
            result.removed = True
            result.error = str(e)

        return result


class AnnotationValidator(BaseCleaner):
    """Validate and fix annotation format (5 values per line)."""

    def clean(self, image_path: Path, label_path: Optional[Path]) -> CleanResult:
        result = CleanResult(str(image_path), str(label_path) if label_path else None)

        if not label_path or not label_path.exists():
            return result

        try:
            valid_lines = []
            modified = False

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        modified = True
                        continue
                    try:
                        int(parts[0])  # class_id
                        [float(x) for x in parts[1:]]  # bbox
                        valid_lines.append(line)
                    except ValueError:
                        modified = True
                        continue

            if modified:
                with open(label_path, 'w') as f:
                    f.writelines(valid_lines)
                result.fixed = True

        except Exception as e:
            result.error = str(e)

        return result


class BBoxValidator(BaseCleaner):
    """Validate bounding box constraints."""

    def __init__(self, min_size: int = 1, max_ratio: float = 0.9):
        """
        Args:
            min_size: Minimum bbox dimension in pixels.
            max_ratio: Maximum bbox size relative to image.
        """
        self.min_size = min_size
        self.max_ratio = max_ratio

    def clean(self, image_path: Path, label_path: Optional[Path]) -> CleanResult:
        result = CleanResult(str(image_path), str(label_path) if label_path else None)

        if not label_path or not label_path.exists():
            return result

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return result

            h, w = img.shape[:2]
            valid_lines = []
            modified = False

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        modified = True
                        continue

                    try:
                        bbox = [float(x) for x in parts[1:]]
                    except ValueError:
                        modified = True
                        continue

                    # Check normalized values
                    if not all(0 <= x <= 1 for x in bbox):
                        modified = True
                        continue

                    # Check size constraints
                    bbox_w, bbox_h = bbox[2], bbox[3]
                    if bbox_w * w < self.min_size or bbox_h * h < self.min_size:
                        modified = True
                        continue
                    if bbox_w > self.max_ratio or bbox_h > self.max_ratio:
                        modified = True
                        continue

                    valid_lines.append(line)

            if modified:
                with open(label_path, 'w') as f:
                    f.writelines(valid_lines)
                result.fixed = True

        except Exception as e:
            result.error = str(e)

        return result
