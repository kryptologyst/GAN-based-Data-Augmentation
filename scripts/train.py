#!/usr/bin/env python3
"""Main training script for GAN-based data augmentation."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf

from src.training.trainer import GANTrainer
from src.training.callbacks import SampleCallback, MetricsCallback
from src.data.datasets import MNISTDataset, create_dataloader
from src.utils.device import get_device, get_device_info
from src.utils.seed import set_seed


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def create_logger(config: Dict[str, Any]) -> pl.loggers.Logger:
    """Create logger based on configuration."""
    if config.get("use_wandb", False):
        return WandbLogger(
            project=config.get("wandb_project", "data-augmentation-gan"),
            name=config.get("experiment_name", "gan_experiment"),
        )
    else:
        return TensorBoardLogger(
            save_dir=config.get("log_dir", "logs"),
            name=config.get("experiment_name", "gan_experiment"),
        )


def create_callbacks(config: Dict[str, Any]) -> list:
    """Create training callbacks."""
    callbacks = []
    
    # Sample callback
    if config.get("sample_frequency", 0) > 0:
        callbacks.append(SampleCallback(
            sample_frequency=config.get("sample_frequency", 5),
            num_samples=config.get("num_samples", 64),
            save_dir=config.get("assets_dir", "assets") + "/generated",
        ))
    
    # Metrics callback
    callbacks.append(MetricsCallback())
    
    return callbacks


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GAN for data augmentation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--fast-dev-run", action="store_true", help="Fast dev run for testing")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config.get("log_level", "INFO"))
    logger = logging.getLogger(__name__)
    
    # Set seed for reproducibility
    set_seed(config.get("seed", 42))
    
    # Get device info
    device = get_device(config.get("device", "auto"))
    device_info = get_device_info()
    logger.info(f"Using device: {device}")
    logger.info(f"Device info: {device_info}")
    
    # Create data loaders
    train_dataset = MNISTDataset(
        data_dir=config.get("data_dir", "data"),
        train=True,
        download=True,
        transforms=config["data"]["train_transforms"],
    )
    
    val_dataset = MNISTDataset(
        data_dir=config.get("data_dir", "data"),
        train=False,
        download=False,
        transforms=config["data"]["val_transforms"],
    )
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        persistent_workers=config["data"]["persistent_workers"],
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        persistent_workers=config["data"]["persistent_workers"],
    )
    
    # Create model
    model = GANTrainer(
        model_config=config["model"],
        loss_config=config["model"]["loss"],
        optimizer_config=config["model"]["optimizer"],
        scheduler_config=config["model"].get("scheduler"),
        z_dim=config["model"]["generator"]["z_dim"],
        lambda_gp=config["model"]["loss"].get("lambda_gp", 10.0),
    )
    
    # Create logger
    pl_logger = create_logger(config)
    
    # Create callbacks
    callbacks = create_callbacks(config)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        logger=pl_logger,
        callbacks=callbacks,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        precision=config["training"]["precision"],
        val_check_interval=config["training"]["val_check_interval"],
        limit_val_batches=config["training"]["limit_val_batches"],
        fast_dev_run=args.fast_dev_run,
        resume_from_checkpoint=args.resume,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    final_checkpoint_path = Path(config.get("checkpoint_dir", "checkpoints")) / "final_model.ckpt"
    trainer.save_checkpoint(str(final_checkpoint_path))
    logger.info(f"Training completed. Final model saved to {final_checkpoint_path}")


if __name__ == "__main__":
    main()
