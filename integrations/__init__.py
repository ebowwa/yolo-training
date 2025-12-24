"""
GPU Provider Integrations

Composable cloud GPU provider integrations for training and inference.
Supports RunPod, Roboflow, Modal (coming soon), and other providers.

Usage:
    from integrations import create_provider, GPUProvider
    from integrations.runpod import RunPodConfig
    from integrations.roboflow import RoboflowConfig, RoboflowProvider
    
    # RunPod for training
    config = RunPodConfig(training_endpoint_id="xxx")
    provider = create_provider("runpod", config)
    
    # Roboflow for preprocessing
    rf_config = RoboflowConfig(model_id="usd-classification/1")
    rf = RoboflowProvider(rf_config)
    result = rf.classify("image.jpg")
"""

from .base import (
    GPUProvider,
    GPUProviderConfig,
    TrainingJobConfig,
    InferenceJobConfig,
    JobState,
    JobStatus,
    JobResult,
)


def create_provider(provider_type: str, config: GPUProviderConfig) -> GPUProvider:
    """
    Factory function to create a GPU provider.
    
    Args:
        provider_type: "runpod" or "modal" (coming soon)
        config: Provider-specific configuration
    """
    if provider_type == "runpod":
        from .runpod import RunPodProvider
        return RunPodProvider(config)
    elif provider_type == "modal":
        raise NotImplementedError("Modal provider coming soon")
    else:
        raise ValueError(f"Unknown provider: {provider_type}. Supported: runpod, modal")


__all__ = [
    # Base classes
    "GPUProvider",
    "GPUProviderConfig",
    "TrainingJobConfig",
    "InferenceJobConfig",
    "JobState",
    "JobStatus",
    "JobResult",
    # Factory
    "create_provider",
]
