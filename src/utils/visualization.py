"""Visualization utilities for generated samples."""

from typing import Optional, Tuple, List
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def save_image_grid(
    images: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    normalize: bool = True,
    title: Optional[str] = None,
) -> None:
    """Save a grid of images.
    
    Args:
        images: Tensor of images (B, C, H, W)
        save_path: Path to save the image
        nrow: Number of images per row
        normalize: Whether to normalize pixel values
        title: Optional title for the plot
    """
    # Convert to numpy and denormalize if needed
    if normalize:
        images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]
    
    images = images.clamp(0, 1).cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(
        nrows=(len(images) + nrow - 1) // nrow,
        ncols=nrow,
        figsize=(nrow * 2, (len(images) + nrow - 1) // nrow * 2)
    )
    
    if len(images) == 1:
        axes = [axes]
    elif len(images) <= nrow:
        axes = axes.reshape(1, -1)
    
    # Plot images
    for i, img in enumerate(images):
        row = i // nrow
        col = i % nrow
        
        if len(images) <= nrow:
            ax = axes[0, col] if len(images) > 1 else axes[col]
        else:
            ax = axes[row, col]
        
        if img.shape[0] == 1:  # Grayscale
            ax.imshow(img[0], cmap='gray')
        else:  # RGB
            ax.imshow(np.transpose(img, (1, 2, 0)))
        
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(images), len(axes.flatten())):
        row = i // nrow
        col = i % nrow
        if len(images) <= nrow:
            ax = axes[0, col] if len(images) > 1 else axes[col]
        else:
            ax = axes[row, col]
        ax.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(
    losses: dict,
    save_path: str,
    title: str = "Training Curves",
) -> None:
    """Plot training loss curves.
    
    Args:
        losses: Dictionary of loss values over time
        save_path: Path to save the plot
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, values in losses.items():
        ax.plot(values, label=name)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_interpolation(
    model: torch.nn.Module,
    z1: torch.Tensor,
    z2: torch.Tensor,
    steps: int = 10,
    save_path: str,
    title: str = "Latent Space Interpolation",
) -> None:
    """Plot interpolation between two points in latent space.
    
    Args:
        model: Trained generator model
        z1: First noise vector
        z2: Second noise vector
        steps: Number of interpolation steps
        save_path: Path to save the plot
        title: Title for the plot
    """
    model.eval()
    with torch.no_grad():
        alphas = torch.linspace(0, 1, steps, device=z1.device)
        interpolated_samples = []
        
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            sample = model(z_interp)
            interpolated_samples.append(sample)
        
        interpolated_samples = torch.cat(interpolated_samples, dim=0)
    
    save_image_grid(
        interpolated_samples,
        save_path,
        nrow=steps,
        title=title,
    )


def plot_latent_traversal(
    model: torch.nn.Module,
    z_base: torch.Tensor,
    dim_idx: int,
    range_vals: Tuple[float, float] = (-3, 3),
    steps: int = 10,
    save_path: str,
    title: str = "Latent Space Traversal",
) -> None:
    """Plot traversal along a single latent dimension.
    
    Args:
        model: Trained generator model
        z_base: Base noise vector
        dim_idx: Dimension to traverse
        range_vals: Range of values to traverse
        steps: Number of steps
        save_path: Path to save the plot
        title: Title for the plot
    """
    model.eval()
    with torch.no_grad():
        z_traversal = z_base.clone()
        traversal_samples = []
        
        for val in np.linspace(range_vals[0], range_vals[1], steps):
            z_traversal[0, dim_idx] = val
            sample = model(z_traversal)
            traversal_samples.append(sample)
        
        traversal_samples = torch.cat(traversal_samples, dim=0)
    
    save_image_grid(
        traversal_samples,
        save_path,
        nrow=steps,
        title=f"{title} - Dim {dim_idx}",
    )
