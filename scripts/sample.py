#!/usr/bin/env python3
"""Sampling script for generating images with trained GAN."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
from omegaconf import OmegaConf

from src.models.dcgan import DCGAN
from src.utils.device import get_device
from src.utils.seed import set_seed
from src.utils.visualization import save_image_grid, plot_interpolation, plot_latent_traversal


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def load_model(checkpoint_path: str, config: Dict[str, Any], device: torch.device) -> DCGAN:
    """Load trained model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove 'model.' prefix from keys
        model_state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
    else:
        model_state_dict = checkpoint
    
    # Create model
    model = DCGAN(
        generator_config=config["model"]["generator"],
        discriminator_config=config["model"]["discriminator"],
    )
    
    # Load state dict
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    return model


def generate_samples(
    model: DCGAN,
    num_samples: int,
    device: torch.device,
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate samples using the trained model."""
    if seed is not None:
        set_seed(seed)
    
    with torch.no_grad():
        samples = model.sample(num_samples, device)
    
    if save_path:
        save_image_grid(
            samples,
            save_path,
            nrow=8,
            title=f"Generated Samples (n={num_samples})",
        )
    
    return samples


def generate_interpolation(
    model: DCGAN,
    device: torch.device,
    save_path: str,
    z_dim: int = 100,
    steps: int = 10,
    seed: Optional[int] = None,
) -> None:
    """Generate interpolation between two random points."""
    if seed is not None:
        set_seed(seed)
    
    # Generate two random noise vectors
    z1 = torch.randn(1, z_dim, device=device)
    z2 = torch.randn(1, z_dim, device=device)
    
    plot_interpolation(
        model.generator,
        z1,
        z2,
        steps=steps,
        save_path=save_path,
        title="Latent Space Interpolation",
    )


def generate_traversal(
    model: DCGAN,
    device: torch.device,
    save_path: str,
    z_dim: int = 100,
    dim_idx: int = 0,
    range_vals: tuple = (-3, 3),
    steps: int = 10,
    seed: Optional[int] = None,
) -> None:
    """Generate latent space traversal."""
    if seed is not None:
        set_seed(seed)
    
    # Generate base noise vector
    z_base = torch.randn(1, z_dim, device=device)
    
    plot_latent_traversal(
        model.generator,
        z_base,
        dim_idx=dim_idx,
        range_vals=range_vals,
        steps=steps,
        save_path=save_path,
        title=f"Latent Space Traversal - Dimension {dim_idx}",
    )


def main() -> None:
    """Main sampling function."""
    parser = argparse.ArgumentParser(description="Generate samples with trained GAN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default="assets/generated", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--interpolation", action="store_true", help="Generate interpolation")
    parser.add_argument("--traversal", action="store_true", help="Generate latent traversal")
    parser.add_argument("--dim", type=int, default=0, help="Dimension for traversal")
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    samples = generate_samples(
        model,
        args.num_samples,
        device,
        save_path=str(output_dir / "samples.png"),
        seed=args.seed,
    )
    
    # Generate interpolation if requested
    if args.interpolation:
        logger.info("Generating interpolation...")
        generate_interpolation(
            model,
            device,
            save_path=str(output_dir / "interpolation.png"),
            z_dim=config["model"]["generator"]["z_dim"],
            seed=args.seed,
        )
    
    # Generate traversal if requested
    if args.traversal:
        logger.info(f"Generating latent traversal for dimension {args.dim}...")
        generate_traversal(
            model,
            device,
            save_path=str(output_dir / f"traversal_dim_{args.dim}.png"),
            z_dim=config["model"]["generator"]["z_dim"],
            dim_idx=args.dim,
            seed=args.seed,
        )
    
    logger.info(f"Generation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
