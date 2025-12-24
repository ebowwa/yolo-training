#!/usr/bin/env python3
"""
RT-DETR Fine-tuning for Cash/Currency Detection.

Downloads dataset from Roboflow and fine-tunes RT-DETR model.
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_roboflow_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    output_dir: str = "datasets/cash",
):
    """
    Download dataset from Roboflow.
    
    Args:
        api_key: Roboflow API key
        workspace: Workspace name
        project: Project name
        version: Dataset version number
        output_dir: Directory to save dataset
        
    Returns:
        Path to downloaded dataset location
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("Roboflow not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "roboflow"])
        from roboflow import Roboflow
    
    logger.info(f"Downloading dataset from Roboflow: {workspace}/{project} v{version}")
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8", location=output_dir)
    
    logger.info(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


def train_rtdetr(
    data_yaml: str,
    model: str = "rtdetr-l.pt",
    epochs: int = 100,
    batch: int = 8,
    imgsz: int = 640,
    device: str = "mps",
    project: str = "runs/rtdetr",
    name: str = "cash_detector",
    patience: int = 20,
    lr0: float = 0.001,
):
    """
    Fine-tune RT-DETR on custom dataset.
    
    Args:
        data_yaml: Path to data.yaml file
        model: Base model to  fine-tune from
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Image size
        device: Device to use (cuda, mps, cpu)
        project: Project directory
        name: Experiment name
        patience: Early stopping patience
        lr0: Initial learning rate
    """
    from ultralytics import RTDETR
    
    logger.info(f"Loading RT-DETR model: {model}")
    model = RTDETR(model)
    
    logger.info(f"Starting training on {data_yaml}")
    logger.info(f"Configuration: epochs={epochs}, batch={batch}, imgsz={imgsz}, device={device}")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        lr0=lr0,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,
    )
    
    logger.info(f"Training complete! Best model saved to: {project}/{name}/weights/best.pt")
    return results


def main():
    parser = argparse.ArgumentParser(description='Train RT-DETR on Roboflow cash dataset')
    
    # Roboflow settings
    parser.add_argument('--api-key', type=str, required=True,
                        help='Roboflow API key')
    parser.add_argument('--workspace', type=str, required=True,
                        help='Roboflow workspace name')
    parser.add_argument('--project', type=str, required=True,
                        help='Roboflow project name')
    parser.add_argument('--version', type=int, default=1,
                        help='Dataset version')
    parser.add_argument('--dataset-dir', type=str, default='datasets/cash',
                        help='Directory to download dataset')
    
    # Training settings
    parser.add_argument('--model', type=str, default='rtdetr-l.pt',
                        choices=['rtdetr-l.pt', 'rtdetr-x.pt'],
                        help='Base RT-DETR model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device (cuda, mps, cpu)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Output settings
    parser.add_argument('--project', type=str, default='runs/rtdetr',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='cash_detector',
                        help='Experiment name')
    
    # Workflow control
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip dataset download if already exists')
    
    args = parser.parse_args()
    
    # Download dataset
    dataset_dir = Path(args.dataset_dir)
    data_yaml = dataset_dir / "data.yaml"
    
    if not args.skip_download or not data_yaml.exists():
        download_roboflow_dataset(
            api_key=args.api_key,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            output_dir=str(dataset_dir),
        )
    else:
        logger.info(f"Using existing dataset at: {dataset_dir}")
    
    # Verify data.yaml exists
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
    
    # Train RT-DETR
    train_rtdetr(
        data_yaml=str(data_yaml),
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        lr0=args.lr,
    )


if __name__ == '__main__':
    main()
