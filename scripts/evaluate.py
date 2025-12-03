#!/usr/bin/env python3
"""Evaluation script for trained GAN models."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from omegaconf import OmegaConf

from src.evaluation.metrics import GANEvaluator
from src.data.datasets import MNISTDataset, create_dataloader
from src.models.dcgan import DCGAN
from src.utils.device import get_device


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def load_model(checkpoint_path: str, config: Dict[str, Any], device: torch.device) -> DCGAN:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
    else:
        model_state_dict = checkpoint
    
    model = DCGAN(
        generator_config=config["model"]["generator"],
        discriminator_config=config["model"]["discriminator"],
    )
    
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    return model


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained GAN model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples for evaluation")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument("--output-dir", type=str, default="assets/evaluation", help="Output directory")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Get device
    device = get_device(config.get("device", "auto"))
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)
    
    # Create data loader
    test_dataset = MNISTDataset(
        data_dir=config.get("data_dir", "data"),
        train=False,
        download=False,
        transforms=config["data"]["val_transforms"],
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Create evaluator
    evaluator = GANEvaluator(device=device, batch_size=args.batch_size)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    logger.info(f"Running evaluation with {args.num_samples} samples...")
    metrics = evaluator.evaluate(
        generator=model.generator,
        real_dataloader=test_loader,
        num_samples=args.num_samples,
        save_path=str(output_dir / "generated_samples.npy"),
    )
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info("=" * 50)
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name.upper()}: {value:.4f}")
    
    # Save results
    import json
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
