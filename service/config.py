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
        
        # Auto-download if missing (inspired by RF-DETR)
        path = ModelRegistry.download_if_missing("yolov8n.pt")
    """
    
    MODELS_DIR = "resources"
    DEFAULT = "yolov8m.pt"
    
    # Hosted models for auto-download (inspired by RF-DETR pattern)
    # URLs for official Ultralytics YOLO models
    HOSTED_MODELS = {
        # YOLOv8 variants
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
        "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
        "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt",
        "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",
        # YOLOv11 variants
        "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "yolo11s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        "yolo11m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        "yolo11l.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
        "yolo11x.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
    }
    
    # RF-DETR model variants (require rfdetr package)
    RFDETR_MODELS = {
        "rfdetr-nano": {"params": "30.5M", "resolution": 384, "coco_ap": 48.4},
        "rfdetr-small": {"params": "32.1M", "resolution": 512, "coco_ap": 53.0},
        "rfdetr-medium": {"params": "33.7M", "resolution": 576, "coco_ap": 54.7},
        "rfdetr-base": {"params": "29M", "resolution": 560, "coco_ap": 53.3},
        "rfdetr-large": {"params": "128M", "resolution": 560, "coco_ap": 56.0},
        "rfdetr-seg": {"params": "~35M", "resolution": 384, "coco_ap": 42.7, "segmentation": True},
    }
    
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
    def list_downloadable(cls) -> List[str]:
        """List all models available for download."""
        return list(cls.HOSTED_MODELS.keys())
    
    @classmethod
    def exists(cls, model_name: str) -> bool:
        """Check if a model exists in the registry."""
        return Path(cls.get_path(model_name)).exists()
    
    @classmethod
    def download_if_missing(cls, model_name: str, force: bool = False) -> str:
        """
        Download a model if it's not present locally.
        
        Inspired by RF-DETR's download_pretrain_weights pattern.
        
        Args:
            model_name: Name of the model to download (e.g., "yolov8n.pt")
            force: If True, re-download even if file exists
            
        Returns:
            Path to the downloaded model
            
        Raises:
            ValueError: If model is not in HOSTED_MODELS
        """
        import logging
        import urllib.request
        
        model_path = cls.get_path(model_name)
        
        # Already exists and not forcing re-download
        if cls.exists(model_name) and not force:
            logging.info(f"Model already exists: {model_path}")
            return model_path
        
        # Check if model is available for download
        if model_name not in cls.HOSTED_MODELS:
            available = ", ".join(cls.HOSTED_MODELS.keys())
            raise ValueError(
                f"Model '{model_name}' not found in HOSTED_MODELS. "
                f"Available: {available}"
            )
        
        # Ensure directory exists
        Path(cls.MODELS_DIR).mkdir(parents=True, exist_ok=True)
        
        url = cls.HOSTED_MODELS[model_name]
        logging.info(f"Downloading {model_name} from {url}...")
        
        try:
            urllib.request.urlretrieve(url, model_path)
            logging.info(f"Successfully downloaded: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download {model_name}: {e}")
        
        return model_path
    
    @classmethod
    def is_rfdetr_model(cls, model_name: str) -> bool:
        """Check if model name is an RF-DETR variant."""
        return model_name.startswith("rfdetr-") or model_name in cls.RFDETR_MODELS
    
    @classmethod
    def get_backend(cls, model_name: str) -> str:
        """
        Determine which backend to use for a given model name.
        
        Returns:
            "rfdetr" or "yolo"
        """
        if cls.is_rfdetr_model(model_name):
            return "rfdetr"
        return "yolo"
    
    @classmethod
    def list_all_models(cls) -> dict:
        """
        List all available models (YOLO + RF-DETR).
        
        Returns:
            Dict with "yolo" and "rfdetr" keys containing model lists
        """
        return {
            "yolo": list(cls.HOSTED_MODELS.keys()),
            "rfdetr": list(cls.RFDETR_MODELS.keys()),
        }


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
