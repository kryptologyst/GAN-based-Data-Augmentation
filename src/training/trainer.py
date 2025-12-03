"""PyTorch Lightning training module for GANs."""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.image import FrechetInceptionDistance, InceptionScore
import numpy as np
from pathlib import Path

from ..models.dcgan import DCGAN
from ..models.losses import GANLoss, compute_gradient_penalty
from ..utils.device import get_device
from ..utils.seed import set_seed


class GANTrainer(pl.LightningModule):
    """PyTorch Lightning module for training GANs.
    
    Args:
        model_config: Configuration for the GAN model
        loss_config: Configuration for the loss function
        optimizer_config: Configuration for optimizers
        scheduler_config: Configuration for schedulers
        z_dim: Dimension of noise vector
        lambda_gp: Weight for gradient penalty
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        loss_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        scheduler_config: Optional[Dict[str, Any]] = None,
        z_dim: int = 100,
        lambda_gp: float = 10.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.z_dim = z_dim
        self.lambda_gp = lambda_gp
        
        # Initialize model
        self.model = DCGAN(
            generator_config=model_config["generator"],
            discriminator_config=model_config["discriminator"],
        )
        
        # Initialize loss function
        self.gan_loss = GANLoss(**loss_config)
        
        # Initialize metrics
        self.metrics = MetricCollection({
            "fid": FrechetInceptionDistance(feature=2048, normalize=True),
            "is": InceptionScore(normalize=True),
        })
        
        # Training step counters
        self.d_steps = 0
        self.g_steps = 0
        
        # Validation samples for consistent evaluation
        self.val_z = None
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from noise."""
        return self.model(z)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step for both generator and discriminator."""
        real_images, _ = batch
        batch_size = real_images.size(0)
        
        # Generate noise
        z = torch.randn(batch_size, self.z_dim, device=self.device)
        
        # Generate fake images
        fake_images = self.model(z)
        
        # Train discriminator
        d_loss = self._discriminator_step(real_images, fake_images)
        
        # Train generator
        g_loss = self._generator_step(fake_images)
        
        # Log losses
        self.log("train/d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/g_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {"d_loss": d_loss, "g_loss": g_loss}
    
    def _discriminator_step(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:
        """Discriminator training step."""
        batch_size = real_images.size(0)
        
        # Real images
        real_pred = self.model.discriminate(real_images)
        real_loss = self.gan_loss(real_pred, target_is_real=True)
        
        # Fake images
        fake_pred = self.model.discriminate(fake_images.detach())
        fake_loss = self.gan_loss(fake_pred, target_is_real=False)
        
        d_loss = real_loss + fake_loss
        
        # Add gradient penalty for WGAN-GP
        if self.gan_loss.gan_mode == "wgangp":
            gp = compute_gradient_penalty(
                self.model.discriminator,
                real_images,
                fake_images.detach(),
                self.device,
            )
            d_loss += self.lambda_gp * gp
        
        # Backward pass
        self.manual_backward(d_loss)
        self.optimizers()[1].step()  # Discriminator optimizer
        self.optimizers()[1].zero_grad()
        
        self.d_steps += 1
        return d_loss
    
    def _generator_step(self, fake_images: torch.Tensor) -> torch.Tensor:
        """Generator training step."""
        fake_pred = self.model.discriminate(fake_images)
        g_loss = self.gan_loss(fake_pred, target_is_real=True)
        
        # Backward pass
        self.manual_backward(g_loss)
        self.optimizers()[0].step()  # Generator optimizer
        self.optimizers()[0].zero_grad()
        
        self.g_steps += 1
        return g_loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        real_images, _ = batch
        
        # Generate validation samples
        if self.val_z is None or self.val_z.size(0) != real_images.size(0):
            self.val_z = torch.randn(real_images.size(0), self.z_dim, device=self.device)
        
        fake_images = self.model(self.val_z)
        
        # Update metrics
        self.metrics.update(fake_images, real=False)
        self.metrics.update(real_images, real=True)
        
        return {"val_loss": torch.tensor(0.0)}  # Placeholder
    
    def on_validation_epoch_end(self) -> None:
        """Compute validation metrics."""
        metrics = self.metrics.compute()
        
        for name, value in metrics.items():
            self.log(f"val/{name}", value, on_epoch=True, prog_bar=True)
        
        self.metrics.reset()
    
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """Configure optimizers for generator and discriminator."""
        g_optimizer = torch.optim.Adam(
            self.model.generator.parameters(),
            **self.hparams.optimizer_config["generator"]
        )
        d_optimizer = torch.optim.Adam(
            self.model.discriminator.parameters(),
            **self.hparams.optimizer_config["discriminator"]
        )
        
        return g_optimizer, d_optimizer
    
    def sample(self, num_samples: int = 64) -> torch.Tensor:
        """Generate samples for visualization."""
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.z_dim, device=self.device)
            samples = self.model(z)
        return samples
    
    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """Interpolate between two noise vectors."""
        self.model.eval()
        with torch.no_grad():
            alphas = torch.linspace(0, 1, steps, device=self.device)
            interpolated_samples = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                sample = self.model(z_interp)
                interpolated_samples.append(sample)
            
            return torch.cat(interpolated_samples, dim=0)
