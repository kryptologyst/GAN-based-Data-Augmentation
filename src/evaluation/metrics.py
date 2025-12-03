"""Evaluation metrics for generative models."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchmetrics import MetricCollection
from torchmetrics.image import FrechetInceptionDistance, InceptionScore
from clean_fid import fid
import lpips
from pathlib import Path


class GANEvaluator:
    """Comprehensive evaluator for GAN models.
    
    Args:
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
    """
    
    def __init__(self, device: torch.device, batch_size: int = 64) -> None:
        self.device = device
        self.batch_size = batch_size
        
        # Initialize metrics
        self.metrics = MetricCollection({
            "fid": FrechetInceptionDistance(feature=2048, normalize=True),
            "is": InceptionScore(normalize=True),
        })
        
        # Initialize LPIPS for diversity measurement
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        
        # Cache for real features
        self.real_features_cache = None
    
    def evaluate(
        self,
        generator: nn.Module,
        real_dataloader: torch.utils.data.DataLoader,
        num_samples: int = 10000,
        save_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """Comprehensive evaluation of the generator.
        
        Args:
            generator: Trained generator model
            real_dataloader: DataLoader for real images
            num_samples: Number of samples to generate for evaluation
            save_path: Optional path to save generated samples
            
        Returns:
            Dictionary of evaluation metrics
        """
        generator.eval()
        
        # Generate samples
        generated_samples = self._generate_samples(generator, num_samples)
        
        # Load real samples
        real_samples = self._load_real_samples(real_dataloader, num_samples)
        
        # Compute metrics
        metrics = {}
        
        # FID and IS
        fid_score = self._compute_fid(generated_samples, real_samples)
        is_score = self._compute_is(generated_samples)
        
        metrics["fid"] = fid_score
        metrics["is"] = is_score
        
        # Precision and Recall
        precision, recall = self._compute_precision_recall(generated_samples, real_samples)
        metrics["precision"] = precision
        metrics["recall"] = recall
        
        # Diversity (LPIPS)
        diversity = self._compute_diversity(generated_samples)
        metrics["diversity"] = diversity
        
        # Save samples if requested
        if save_path:
            self._save_samples(generated_samples, save_path)
        
        return metrics
    
    def _generate_samples(self, generator: nn.Module, num_samples: int) -> torch.Tensor:
        """Generate samples using the generator."""
        samples = []
        
        with torch.no_grad():
            for _ in range(0, num_samples, self.batch_size):
                batch_size = min(self.batch_size, num_samples - len(samples))
                z = torch.randn(batch_size, generator.z_dim, device=self.device)
                batch_samples = generator(z)
                samples.append(batch_samples)
        
        return torch.cat(samples, dim=0)[:num_samples]
    
    def _load_real_samples(self, dataloader: torch.utils.data.DataLoader, num_samples: int) -> torch.Tensor:
        """Load real samples from the dataloader."""
        samples = []
        
        for batch, _ in dataloader:
            samples.append(batch.to(self.device))
            if len(torch.cat(samples, dim=0)) >= num_samples:
                break
        
        return torch.cat(samples, dim=0)[:num_samples]
    
    def _compute_fid(self, generated: torch.Tensor, real: torch.Tensor) -> float:
        """Compute Fréchet Inception Distance."""
        # Convert to numpy arrays
        generated_np = generated.cpu().numpy()
        real_np = real.cpu().numpy()
        
        # Convert from [-1, 1] to [0, 1]
        generated_np = (generated_np + 1) / 2
        real_np = (real_np + 1) / 2
        
        # Convert to uint8
        generated_np = (generated_np * 255).astype(np.uint8)
        real_np = (real_np * 255).astype(np.uint8)
        
        # Compute FID using clean-fid
        fid_score = fid.compute_fid(
            real_np,
            generated_np,
            mode="clean",
            device=self.device,
        )
        
        return fid_score
    
    def _compute_is(self, generated: torch.Tensor) -> float:
        """Compute Inception Score."""
        # Update metrics
        self.metrics.update(generated, real=False)
        is_score = self.metrics["is"].compute().item()
        self.metrics.reset()
        
        return is_score
    
    def _compute_precision_recall(
        self,
        generated: torch.Tensor,
        real: torch.Tensor,
    ) -> Tuple[float, float]:
        """Compute Precision and Recall metrics."""
        # Extract features using a pre-trained model
        generated_features = self._extract_features(generated)
        real_features = self._extract_features(real)
        
        # Compute pairwise distances
        generated_distances = self._compute_pairwise_distances(generated_features)
        real_distances = self._compute_pairwise_distances(real_features)
        
        # Compute precision and recall
        precision = self._compute_precision(generated_distances, real_distances)
        recall = self._compute_recall(generated_distances, real_distances)
        
        return precision, recall
    
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features using a pre-trained model."""
        # Use a simple CNN for feature extraction
        # In practice, you might want to use a pre-trained model
        features = []
        
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            # Simple feature extraction - in practice use a pre-trained model
            batch_features = F.adaptive_avg_pool2d(batch, (1, 1)).view(batch.size(0), -1)
            features.append(batch_features)
        
        return torch.cat(features, dim=0)
    
    def _compute_pairwise_distances(self, features: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between features."""
        # Compute L2 distances
        distances = torch.cdist(features, features, p=2)
        return distances
    
    def _compute_precision(
        self,
        generated_distances: torch.Tensor,
        real_distances: torch.Tensor,
    ) -> float:
        """Compute precision metric."""
        # Simplified precision computation
        # In practice, you'd use more sophisticated methods
        return 0.5  # Placeholder
    
    def _compute_recall(
        self,
        generated_distances: torch.Tensor,
        real_distances: torch.Tensor,
    ) -> float:
        """Compute recall metric."""
        # Simplified recall computation
        # In practice, you'd use more sophisticated methods
        return 0.5  # Placeholder
    
    def _compute_diversity(self, generated: torch.Tensor) -> float:
        """Compute diversity using LPIPS."""
        diversity_scores = []
        
        with torch.no_grad():
            for i in range(0, len(generated) - 1, self.batch_size):
                batch1 = generated[i:i + self.batch_size]
                batch2 = generated[i + 1:i + 1 + self.batch_size]
                
                if len(batch1) == len(batch2):
                    # Convert to RGB if needed
                    if batch1.size(1) == 1:
                        batch1 = batch1.repeat(1, 3, 1, 1)
                        batch2 = batch2.repeat(1, 3, 1, 1)
                    
                    # Compute LPIPS distances
                    distances = self.lpips_model(batch1, batch2)
                    diversity_scores.append(distances.mean().item())
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _save_samples(self, samples: torch.Tensor, save_path: str) -> None:
        """Save generated samples."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy and save
        samples_np = samples.cpu().numpy()
        np.save(save_path, samples_np)
        
        # Also save a visualization
        from ..utils.visualization import save_image_grid
        viz_path = save_path.with_suffix('.png')
        save_image_grid(
            samples[:64],  # Save first 64 samples
            str(viz_path),
            nrow=8,
            title="Generated Samples for Evaluation",
        )
