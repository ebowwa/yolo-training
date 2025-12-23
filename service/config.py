"""
Configuration dataclasses for YOLO Training Service.
Type-safe configuration for all service operations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


class ModelRegistry:
    """
    Centralized registry for YOLO model paths.
    
    All model path references should go through this registry to ensure
    consistency and make model management easier.
    
    Usage:
        from service.config import ModelRegistry
        
        # Get a specific model
        path = ModelRegistry.get_path("yolov8m.pt")  # -> "resources/yolov8m.pt"
        
        # Use the default model
        path = ModelRegistry.get_path(ModelRegistry.DEFAULT)
        
        # List available models
        models = ModelRegistry.list_available()
    """
    
    MODELS_DIR = "resources"
    DEFAULT = "yolov8m.pt"
    
    @classmethod
    def get_path(cls, model_name: str) -> str:
        """
        Get the full path for a model by name.
        
        Args:
            model_name: Name of the model file (e.g., "yolov8m.pt")
            
        Returns:
            Full path to the model (e.g., "resources/yolov8m.pt")
        """
        # If already a full path, return as-is
        if "/" in model_name or "\\" in model_name:
            return model_name
        return f"{cls.MODELS_DIR}/{model_name}"
    
    @classmethod
    def get_default_path(cls) -> str:
        """Get the path to the default model."""
        return cls.get_path(cls.DEFAULT)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available .pt model files in the models directory.
        
        Returns:
            List of model filenames (e.g., ["yolov8m.pt", "pothole.pt"])
        """
        models_path = Path(cls.MODELS_DIR)
        if not models_path.exists():
            return []
        return [f.name for f in models_path.glob("*.pt")]
    
    @classmethod
    def exists(cls, model_name: str) -> bool:
        """Check if a model exists in the registry."""
        return Path(cls.get_path(model_name)).exists()


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
    base_model: str = None  # Base model, defaults to ModelRegistry.DEFAULT
    
    def __post_init__(self):
        if self.base_model is None:
            self.base_model = ModelRegistry.get_default_path()


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
