"""
RF-DETR Service - Alternative transformer-based detection backend.

Wraps RF-DETR models (from Roboflow) as an alternative to YOLO
for higher accuracy object detection at the cost of speed.

RF-DETR achieves state-of-the-art accuracy on COCO (60+ AP) while
maintaining real-time performance.

Requires: pip install rfdetr
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# RF-DETR is optional - gracefully handle if not installed
try:
    from rfdetr import (
        RFDETRBase,
        RFDETRLarge,
        RFDETRNano,
        RFDETRSmall,
        RFDETRMedium,
        RFDETRSegPreview,
    )
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    RFDETRBase = None
    RFDETRLarge = None
    RFDETRNano = None
    RFDETRSmall = None
    RFDETRMedium = None
    RFDETRSegPreview = None


# Model variant mapping
RFDETR_MODELS = {
    "rfdetr-nano": RFDETRNano,
    "rfdetr-small": RFDETRSmall,
    "rfdetr-medium": RFDETRMedium,
    "rfdetr-base": RFDETRBase,
    "rfdetr-large": RFDETRLarge,
    "rfdetr-seg": RFDETRSegPreview,
}


@dataclass
class RFDETRDetection:
    """Single detection result from RF-DETR."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    mask: Optional[np.ndarray] = None  # For segmentation


@dataclass
class RFDETRResult:
    """Result from RF-DETR inference."""
    detections: List[RFDETRDetection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    image_size: tuple = (0, 0)


class RFDETRService:
    """
    Service wrapper for RF-DETR object detection models.
    
    Provides a unified interface matching the YOLO InferenceService pattern,
    making it easy to swap between YOLO and RF-DETR backends.
    
    Usage:
        from service.rfdetr_service import RFDETRService
        
        # Create service with a model variant
        service = RFDETRService("rfdetr-medium")
        
        # Run inference
        result = service.infer_image("path/to/image.jpg", conf_threshold=0.5)
        
        for det in result.detections:
            print(f"{det.class_name}: {det.confidence:.2f}")
    """
    
    # Model benchmarks (COCO mAP50-95)
    MODEL_INFO = {
        "rfdetr-nano": {"params": "30.5M", "resolution": 384, "coco_ap": 48.4},
        "rfdetr-small": {"params": "32.1M", "resolution": 512, "coco_ap": 53.0},
        "rfdetr-medium": {"params": "33.7M", "resolution": 576, "coco_ap": 54.7},
        "rfdetr-base": {"params": "29M", "resolution": 560, "coco_ap": 53.3},
        "rfdetr-large": {"params": "128M", "resolution": 560, "coco_ap": 56.0},
        "rfdetr-seg": {"params": "~35M", "resolution": 384, "coco_ap": 42.7},
    }
    
    def __init__(
        self,
        model_name: str = "rfdetr-medium",
        device: str = "auto",
        optimize: bool = True,
    ):
        """
        Initialize RF-DETR service.
        
        Args:
            model_name: Model variant (rfdetr-nano, rfdetr-small, rfdetr-medium, 
                       rfdetr-base, rfdetr-large, rfdetr-seg)
            device: Device to use ("auto", "cuda", "cpu", "mps")
            optimize: Whether to optimize for inference (2x speedup)
        """
        if not RFDETR_AVAILABLE:
            raise ImportError(
                "RF-DETR is not installed. Install with: pip install rfdetr"
            )
        
        if model_name not in RFDETR_MODELS:
            available = ", ".join(RFDETR_MODELS.keys())
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {available}"
            )
        
        self.model_name = model_name
        self.model_class = RFDETR_MODELS[model_name]
        
        logging.info(f"Loading {model_name}...")
        self.model = self.model_class()
        
        if optimize:
            logging.info("Optimizing for inference...")
            self.model.optimize_for_inference()
        
        self._is_optimized = optimize
        logging.info(f"RF-DETR {model_name} ready")
    
    @staticmethod
    def is_available() -> bool:
        """Check if RF-DETR is installed."""
        return RFDETR_AVAILABLE
    
    @staticmethod
    def list_models() -> List[str]:
        """List available RF-DETR model variants."""
        return list(RFDETR_MODELS.keys())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        info = self.MODEL_INFO.get(self.model_name, {})
        return {
            "name": self.model_name,
            "optimized": self._is_optimized,
            **info
        }
    
    def infer_image(
        self,
        image: Union[str, np.ndarray, "PIL.Image.Image"],
        conf_threshold: float = 0.5,
    ) -> RFDETRResult:
        """
        Run inference on a single image.
        
        Args:
            image: Image path, numpy array (RGB), or PIL Image
            conf_threshold: Minimum confidence threshold
            
        Returns:
            RFDETRResult with detections
        """
        import time
        
        start = time.perf_counter()
        
        # RF-DETR returns supervision.Detections
        sv_detections = self.model.predict(image, threshold=conf_threshold)
        
        inference_time = (time.perf_counter() - start) * 1000
        
        # Convert to our format
        detections = []
        class_names = self.model.class_names
        
        for i in range(len(sv_detections)):
            class_id = int(sv_detections.class_id[i])
            class_name = class_names.get(class_id, f"class_{class_id}")
            
            det = RFDETRDetection(
                class_id=class_id,
                class_name=class_name,
                confidence=float(sv_detections.confidence[i]),
                bbox=sv_detections.xyxy[i].tolist(),
                mask=sv_detections.mask[i] if sv_detections.mask is not None else None,
            )
            detections.append(det)
        
        return RFDETRResult(
            detections=detections,
            inference_time_ms=inference_time,
        )
    
    def infer_batch(
        self,
        images: List[Union[str, np.ndarray]],
        conf_threshold: float = 0.5,
    ) -> List[RFDETRResult]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of image paths or numpy arrays
            conf_threshold: Minimum confidence threshold
            
        Returns:
            List of RFDETRResult objects
        """
        import time
        
        start = time.perf_counter()
        
        # RF-DETR supports batch inference
        sv_detections_list = self.model.predict(images, threshold=conf_threshold)
        
        if not isinstance(sv_detections_list, list):
            sv_detections_list = [sv_detections_list]
        
        inference_time = (time.perf_counter() - start) * 1000
        per_image_time = inference_time / len(images)
        
        results = []
        class_names = self.model.class_names
        
        for sv_detections in sv_detections_list:
            detections = []
            for i in range(len(sv_detections)):
                class_id = int(sv_detections.class_id[i])
                class_name = class_names.get(class_id, f"class_{class_id}")
                
                det = RFDETRDetection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(sv_detections.confidence[i]),
                    bbox=sv_detections.xyxy[i].tolist(),
                    mask=sv_detections.mask[i] if sv_detections.mask is not None else None,
                )
                detections.append(det)
            
            results.append(RFDETRResult(
                detections=detections,
                inference_time_ms=per_image_time,
            ))
        
        return results
    
    def train(
        self,
        dataset_dir: str,
        epochs: int = 50,
        lr: float = 1e-4,
        batch_size: int = 4,
        output_dir: str = "runs/rfdetr",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Fine-tune RF-DETR on a custom dataset.
        
        Args:
            dataset_dir: Path to dataset in COCO format
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size
            output_dir: Output directory for weights
            **kwargs: Additional training arguments
            
        Returns:
            Training results dict
        """
        logging.info(f"Starting RF-DETR training for {epochs} epochs...")
        
        self.model.train(
            dataset_dir=dataset_dir,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            output_dir=output_dir,
            **kwargs,
        )
        
        return {
            "model": self.model_name,
            "epochs": epochs,
            "output_dir": output_dir,
        }
    
    def export(
        self,
        output_dir: str = "exports",
        format: str = "onnx",
        **kwargs,
    ) -> str:
        """
        Export model to deployment format.
        
        Args:
            output_dir: Output directory
            format: Export format ("onnx")
            **kwargs: Additional export arguments
            
        Returns:
            Path to exported model
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.model.export(
            output_dir=output_dir,
            **kwargs,
        )
        
        return output_dir


def is_rfdetr_model(model_name: str) -> bool:
    """Check if model name is an RF-DETR variant."""
    return model_name.startswith("rfdetr-")


def get_backend_for_model(model_name: str) -> str:
    """
    Determine which backend to use for a given model name.
    
    Returns:
        "rfdetr" or "yolo"
    """
    if is_rfdetr_model(model_name):
        return "rfdetr"
    return "yolo"
