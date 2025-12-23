"""
Configuration dataclasses for YOLO Training Service.
Type-safe configuration for all service operations.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetConfig:
    """Configuration for dataset operations."""
    dataset_handle: str  # Kaggle dataset handle
    nc: int  # Number of classes
    names: List[str]  # Class names
    cache_dir: Optional[str] = None  # Optional custom cache directory


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    epochs: int = 60
    imgsz: int = 512
    batch: int = 32
    device: str = "0"  # GPU index or "cpu"
    project: str = "runs/train"
    name: str = "yolo_train"
    weights: Optional[str] = None  # Custom pretrained weights path
    base_model: str = "yolov8m.pt"  # Base model to use


@dataclass
class InferenceConfig:
    """Configuration for inference operations."""
    conf_threshold: float = 0.5
    save_output: bool = False
    output_path: Optional[str] = None
    iou_threshold: float = 0.45  # NMS IoU threshold


@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    imgsz: int = 512
    split: str = "test"  # Dataset split to validate on
    project: str = "runs/val"
    name: str = "validation"


@dataclass
class ExportConfig:
    """Configuration for model export."""
    format: str = "ncnn"  # Export format: ncnn, onnx, torchscript, etc.
    output_dir: Optional[str] = None
    imgsz: int = 512
    half: bool = False  # FP16 quantization
    dynamic: bool = False  # Dynamic axes for ONNX


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    config_path: Optional[str] = None  # Path to preprocessing config YAML
    augment_factor: int = 2  # Number of augmented versions per image
    clean: bool = True  # Run cleaning before augmentation


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixels
    bbox_normalized: List[float]  # [x_center, y_center, width, height] normalized


@dataclass
class InferenceResult:
    """Result from inference operation."""
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    image_size: tuple = (0, 0)  # (width, height)
    annotated_image: Optional[any] = None  # numpy array if requested


@dataclass
class TrainingResult:
    """Result from training operation."""
    best_model_path: str
    last_model_path: str
    metrics: dict = field(default_factory=dict)
    epochs_completed: int = 0


@dataclass
class ValidationResult:
    """Result from validation operation."""
    map50: float = 0.0  # mAP@0.5
    map50_95: float = 0.0  # mAP@0.5:0.95
    precision: float = 0.0
    recall: float = 0.0
    metrics: dict = field(default_factory=dict)
