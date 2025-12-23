"""
Training callbacks system.

Provides extensible callbacks for training hooks inspired by RF-DETR's
callbacks pattern. Supports TensorBoard, W&B, early stopping, and custom hooks.
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# Optional dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import wandb
except ImportError:
    wandb = None


class TrainingCallback(ABC):
    """
    Base class for training callbacks.
    
    Subclasses should implement the hooks they need.
    All hooks receive a `metrics` dict with training state.
    """
    
    def on_train_start(self, metrics: Dict[str, Any]) -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, metrics: Dict[str, Any]) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_start(self, metrics: Dict[str, Any]) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, metrics: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_end(self, metrics: Dict[str, Any]) -> None:
        """Called at the end of each batch."""
        pass


class CallbackManager:
    """
    Manages a collection of training callbacks.
    
    Usage:
        from service.callbacks import CallbackManager, TensorBoardCallback
        
        manager = CallbackManager()
        manager.add(TensorBoardCallback("runs/experiment1"))
        
        # During training
        manager.on_epoch_end({"epoch": 1, "loss": 0.5, "mAP50": 0.85})
    """
    
    def __init__(self):
        self.callbacks: List[TrainingCallback] = []
    
    def add(self, callback: TrainingCallback) -> "CallbackManager":
        """Add a callback. Returns self for chaining."""
        self.callbacks.append(callback)
        return self
    
    def remove(self, callback: TrainingCallback) -> None:
        """Remove a callback."""
        self.callbacks.remove(callback)
    
    def on_train_start(self, metrics: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_start(metrics)
    
    def on_train_end(self, metrics: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_end(metrics)
    
    def on_epoch_start(self, metrics: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(metrics)
    
    def on_epoch_end(self, metrics: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(metrics)
    
    def on_batch_end(self, metrics: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(metrics)


class TensorBoardCallback(TrainingCallback):
    """
    Logs training metrics to TensorBoard.
    
    Inspired by RF-DETR's MetricsTensorBoardSink.
    
    Args:
        log_dir: Directory for TensorBoard logs
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = None
        
        if SummaryWriter is None:
            logging.warning(
                "TensorBoard not available. Install with: pip install tensorboard"
            )
        else:
            self.writer = SummaryWriter(log_dir=log_dir)
            logging.info(
                f"TensorBoard logging initialized. "
                f"Run: tensorboard --logdir {log_dir}"
            )
    
    def on_epoch_end(self, metrics: Dict[str, Any]) -> None:
        if self.writer is None:
            return
        
        epoch = metrics.get("epoch", 0)
        
        # Log common metrics
        for key in ["loss", "train_loss", "val_loss", "box_loss", "cls_loss"]:
            if key in metrics:
                self.writer.add_scalar(f"Loss/{key}", metrics[key], epoch)
        
        for key in ["mAP50", "mAP50-95", "precision", "recall"]:
            if key in metrics:
                self.writer.add_scalar(f"Metrics/{key}", metrics[key], epoch)
        
        if "lr" in metrics:
            self.writer.add_scalar("Training/learning_rate", metrics["lr"], epoch)
        
        self.writer.flush()
    
    def on_train_end(self, metrics: Dict[str, Any]) -> None:
        if self.writer:
            self.writer.close()


class WandBCallback(TrainingCallback):
    """
    Logs training metrics to Weights & Biases.
    
    Inspired by RF-DETR's MetricsWandBSink.
    
    Args:
        project: W&B project name
        run_name: Name for this run (optional)
        config: Training config to log (optional)
    """
    
    def __init__(
        self,
        project: str,
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        self.project = project
        self.run_name = run_name
        self.run = None
        
        if wandb is None:
            logging.warning(
                "W&B not available. Install with: pip install wandb"
            )
        else:
            self.run = wandb.init(
                project=project,
                name=run_name,
                config=config,
            )
            logging.info(f"W&B logging initialized: {wandb.run.url}")
    
    def on_epoch_end(self, metrics: Dict[str, Any]) -> None:
        if self.run is None or wandb is None:
            return
        
        # W&B handles epoch automatically via step
        log_dict = {}
        
        for key in ["loss", "train_loss", "val_loss", "mAP50", "mAP50-95", 
                    "precision", "recall", "lr"]:
            if key in metrics:
                log_dict[key] = metrics[key]
        
        if log_dict:
            wandb.log(log_dict, step=metrics.get("epoch", None))
    
    def on_train_end(self, metrics: Dict[str, Any]) -> None:
        if self.run:
            self.run.finish()


class EarlyStoppingCallback(TrainingCallback):
    """
    Stops training when a metric stops improving.
    
    Inspired by RF-DETR's EarlyStoppingCallback.
    
    Args:
        monitor: Metric to monitor (default: "mAP50")
        patience: Epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: "max" for metrics to maximize, "min" to minimize
        stop_training_fn: Callable to invoke when stopping (optional)
    """
    
    def __init__(
        self,
        monitor: str = "mAP50",
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
        stop_training_fn: Optional[Callable] = None,
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.stop_training_fn = stop_training_fn
        
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False
    
    def _is_improvement(self, current: float) -> bool:
        if self.mode == "max":
            return current > self.best_value + self.min_delta
        else:
            return current < self.best_value - self.min_delta
    
    def on_epoch_end(self, metrics: Dict[str, Any]) -> None:
        if self.monitor not in metrics:
            logging.warning(
                f"EarlyStopping: metric '{self.monitor}' not found in metrics"
            )
            return
        
        current = metrics[self.monitor]
        epoch = metrics.get("epoch", "?")
        
        if self._is_improvement(current):
            self.best_value = current
            self.counter = 0
            logging.info(
                f"EarlyStopping: {self.monitor} improved to {current:.4f}"
            )
        else:
            self.counter += 1
            logging.info(
                f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs "
                f"(best: {self.best_value:.4f}, current: {current:.4f})"
            )
        
        if self.counter >= self.patience:
            self.should_stop = True
            logging.warning(
                f"EarlyStopping triggered at epoch {epoch}: "
                f"No improvement in {self.monitor} for {self.patience} epochs"
            )
            if self.stop_training_fn:
                self.stop_training_fn()


class MetricsLoggerCallback(TrainingCallback):
    """
    Logs metrics to console and optionally saves to file.
    
    Args:
        output_dir: Directory to save metrics JSON (optional)
        log_every_n_epochs: Log every N epochs (default: 1)
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        log_every_n_epochs: int = 1,
    ):
        self.output_dir = output_dir
        self.log_every_n_epochs = log_every_n_epochs
        self.history: List[Dict[str, Any]] = []
    
    def on_epoch_end(self, metrics: Dict[str, Any]) -> None:
        self.history.append(metrics.copy())
        
        epoch = metrics.get("epoch", 0)
        if epoch % self.log_every_n_epochs == 0:
            # Format key metrics for logging
            parts = [f"Epoch {epoch}"]
            for key in ["loss", "mAP50", "mAP50-95", "precision", "recall"]:
                if key in metrics:
                    parts.append(f"{key}={metrics[key]:.4f}")
            logging.info(" | ".join(parts))
    
    def on_train_end(self, metrics: Dict[str, Any]) -> None:
        if self.output_dir:
            import json
            output_path = Path(self.output_dir) / "training_metrics.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(self.history, f, indent=2)
            logging.info(f"Metrics saved to {output_path}")


def create_default_callbacks(
    output_dir: str,
    tensorboard: bool = True,
    wandb_project: Optional[str] = None,
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
) -> CallbackManager:
    """
    Create a CallbackManager with common callbacks preconfigured.
    
    Args:
        output_dir: Directory for logs and outputs
        tensorboard: Enable TensorBoard logging
        wandb_project: W&B project name (enables W&B if set)
        early_stopping: Enable early stopping
        early_stopping_patience: Epochs to wait before stopping
        
    Returns:
        Configured CallbackManager
    """
    manager = CallbackManager()
    
    # Always add metrics logger
    manager.add(MetricsLoggerCallback(output_dir=output_dir))
    
    if tensorboard:
        manager.add(TensorBoardCallback(log_dir=output_dir))
    
    if wandb_project:
        manager.add(WandBCallback(project=wandb_project))
    
    if early_stopping:
        manager.add(EarlyStoppingCallback(patience=early_stopping_patience))
    
    return manager
