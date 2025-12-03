"""Loss functions for GAN training."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class GANLoss(nn.Module):
    """GAN loss function supporting multiple loss types.
    
    Args:
        gan_mode: Type of GAN loss ('vanilla', 'lsgan', 'wgangp')
        target_real_label: Target label for real images
        target_fake_label: Target label for fake images
        lambda_gp: Weight for gradient penalty (for WGAN-GP)
    """
    
    def __init__(
        self,
        gan_mode: str = "vanilla",
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
        lambda_gp: float = 10.0,
    ) -> None:
        super().__init__()
        self.gan_mode = gan_mode
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        self.lambda_gp = lambda_gp
        
        if gan_mode == "vanilla":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif gan_mode == "lsgan":
            self.loss_fn = nn.MSELoss()
        elif gan_mode == "wgangp":
            self.loss_fn = None  # WGAN-GP doesn't use a separate loss function
        else:
            raise ValueError(f"Unsupported GAN mode: {gan_mode}")
    
    def forward(
        self,
        prediction: torch.Tensor,
        target_is_real: bool,
        lambda_gp: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute GAN loss.
        
        Args:
            prediction: Discriminator output
            target_is_real: Whether the target is real or fake
            lambda_gp: Override gradient penalty weight
            
        Returns:
            Computed loss
        """
        if lambda_gp is None:
            lambda_gp = self.lambda_gp
            
        if self.gan_mode == "vanilla":
            if target_is_real:
                target = torch.ones_like(prediction) * self.target_real_label
            else:
                target = torch.zeros_like(prediction) * self.target_fake_label
            return self.loss_fn(prediction, target)
        
        elif self.gan_mode == "lsgan":
            if target_is_real:
                target = torch.ones_like(prediction) * self.target_real_label
            else:
                target = torch.zeros_like(prediction) * self.target_fake_label
            return self.loss_fn(prediction, target)
        
        elif self.gan_mode == "wgangp":
            if target_is_real:
                return -prediction.mean()
            else:
                return prediction.mean()
        
        else:
            raise ValueError(f"Unsupported GAN mode: {self.gan_mode}")


def compute_gradient_penalty(
    discriminator: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute gradient penalty for WGAN-GP.
    
    Args:
        discriminator: Discriminator model
        real_samples: Real samples
        fake_samples: Fake samples
        device: Device to compute on
        
    Returns:
        Gradient penalty loss
    """
    batch_size = real_samples.size(0)
    
    # Random interpolation between real and fake samples
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # Discriminator output for interpolated samples
    d_interpolated = discriminator(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for generator training.
    
    This loss encourages the generator to produce features that match
    the statistics of real features in the discriminator.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(
        self,
        real_features: torch.Tensor,
        fake_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute feature matching loss.
        
        Args:
            real_features: Features from real images
            fake_features: Features from fake images
            
        Returns:
            Feature matching loss
        """
        return self.l1_loss(fake_features, real_features.detach())


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained VGG features.
    
    Args:
        feature_layers: Which VGG layers to use for features
        weights: Weights for each feature layer
    """
    
    def __init__(
        self,
        feature_layers: list = [0, 5, 10, 19, 28],
        weights: list = [1.0, 1.0, 1.0, 1.0, 1.0],
    ) -> None:
        super().__init__()
        self.feature_layers = feature_layers
        self.weights = weights
        
        # Load pre-trained VGG19
        vgg = torchvision.models.vgg19(pretrained=True).features
        self.vgg_layers = nn.ModuleList([vgg[i] for i in feature_layers])
        
        # Freeze VGG parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss.
        
        Args:
            real_images: Real images
            fake_images: Generated images
            
        Returns:
            Perceptual loss
        """
        # Normalize images to VGG input range
        real_images = F.interpolate(real_images, size=(224, 224), mode='bilinear', align_corners=False)
        fake_images = F.interpolate(fake_images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Convert grayscale to RGB if needed
        if real_images.size(1) == 1:
            real_images = real_images.repeat(1, 3, 1, 1)
            fake_images = fake_images.repeat(1, 3, 1, 1)
        
        # Extract features
        real_features = []
        fake_features = []
        
        x_real = real_images
        x_fake = fake_images
        
        for layer in self.vgg_layers:
            x_real = layer(x_real)
            x_fake = layer(x_fake)
            real_features.append(x_real)
            fake_features.append(x_fake)
        
        # Compute perceptual loss
        loss = 0.0
        for real_feat, fake_feat, weight in zip(real_features, fake_features, self.weights):
            loss += weight * F.mse_loss(fake_feat, real_feat)
        
        return loss
