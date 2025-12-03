"""Custom callbacks for GAN training."""

from typing import Any, Dict, Optional
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pathlib import Path

from ..utils.visualization import save_image_grid, plot_interpolation


class SampleCallback(Callback):
    """Callback to generate and save sample images during training.
    
    Args:
        sample_frequency: How often to generate samples (in epochs)
        num_samples: Number of samples to generate
        save_dir: Directory to save samples
    """
    
    def __init__(
        self,
        sample_frequency: int = 5,
        num_samples: int = 64,
        save_dir: str = "assets/generated",
    ) -> None:
        super().__init__()
        self.sample_frequency = sample_frequency
        self.num_samples = num_samples
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Generate samples at the end of training epochs."""
        if (trainer.current_epoch + 1) % self.sample_frequency == 0:
            # Generate samples
            samples = pl_module.sample(self.num_samples)
            
            # Save sample grid
            save_path = self.save_dir / f"samples_epoch_{trainer.current_epoch + 1:03d}.png"
            save_image_grid(
                samples,
                str(save_path),
                nrow=8,
                title=f"Generated Samples - Epoch {trainer.current_epoch + 1}",
            )
            
            # Generate interpolation if we have enough samples
            if self.num_samples >= 2:
                z1 = torch.randn(1, pl_module.z_dim, device=pl_module.device)
                z2 = torch.randn(1, pl_module.z_dim, device=pl_module.device)
                
                interp_path = self.save_dir / f"interpolation_epoch_{trainer.current_epoch + 1:03d}.png"
                plot_interpolation(
                    pl_module.model.generator,
                    z1,
                    z2,
                    steps=10,
                    save_path=str(interp_path),
                    title=f"Latent Interpolation - Epoch {trainer.current_epoch + 1}",
                )


class MetricsCallback(Callback):
    """Callback to log additional metrics during training."""
    
    def __init__(self) -> None:
        super().__init__()
        self.losses = {"d_loss": [], "g_loss": []}
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log epoch-level metrics."""
        # Log learning rates
        g_lr = trainer.optimizers[0].param_groups[0]["lr"]
        d_lr = trainer.optimizers[1].param_groups[0]["lr"]
        
        pl_module.log("train/g_lr", g_lr, on_epoch=True)
        pl_module.log("train/d_lr", d_lr, on_epoch=True)
        
        # Log training step ratios
        pl_module.log("train/d_steps", pl_module.d_steps, on_epoch=True)
        pl_module.log("train/g_steps", pl_module.g_steps, on_epoch=True)
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save final samples and model info."""
        # Generate final samples
        final_samples = pl_module.sample(100)
        save_path = Path("assets/generated/final_samples.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_image_grid(
            final_samples,
            str(save_path),
            nrow=10,
            title="Final Generated Samples",
        )
        
        # Save model info
        model_info = {
            "epochs_trained": trainer.current_epoch + 1,
            "d_steps": pl_module.d_steps,
            "g_steps": pl_module.g_steps,
            "z_dim": pl_module.z_dim,
            "lambda_gp": pl_module.lambda_gp,
        }
        
        import json
        with open("assets/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
