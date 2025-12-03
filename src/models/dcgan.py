"""Modern GAN models for data augmentation."""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np


class Generator(nn.Module):
    """DCGAN Generator with modern improvements.
    
    Args:
        z_dim: Dimension of the noise vector
        hidden_dim: Hidden dimension size
        image_size: Size of generated images (assumes square)
        channels: Number of output channels
    """
    
    def __init__(
        self,
        z_dim: int = 100,
        hidden_dim: int = 64,
        image_size: int = 28,
        channels: int = 1,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.channels = channels
        
        # Calculate the starting size for transposed convolutions
        # For 28x28 output, we start with 7x7 feature maps
        self.start_size = image_size // 4
        
        # Project noise to initial feature map
        self.fc = nn.Linear(z_dim, hidden_dim * 8 * self.start_size * self.start_size)
        
        # Transposed convolution layers
        self.conv1 = nn.ConvTranspose2d(
            hidden_dim * 8, hidden_dim * 4, 
            kernel_size=4, stride=2, padding=1, bias=False
        )
        self.conv2 = nn.ConvTranspose2d(
            hidden_dim * 4, hidden_dim * 2,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        self.conv3 = nn.ConvTranspose2d(
            hidden_dim * 2, hidden_dim,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        self.conv4 = nn.ConvTranspose2d(
            hidden_dim, channels,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(hidden_dim * 4)
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the generator.
        
        Args:
            z: Noise tensor of shape (batch_size, z_dim)
            
        Returns:
            Generated images of shape (batch_size, channels, image_size, image_size)
        """
        batch_size = z.size(0)
        
        # Project noise to feature map
        x = self.fc(z)
        x = x.view(batch_size, self.hidden_dim * 8, self.start_size, self.start_size)
        
        # Apply transposed convolutions with batch norm and ReLU
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.tanh(self.conv4(x))
        
        return x
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample random images from the generator.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated images
        """
        z = torch.randn(num_samples, self.z_dim, device=device)
        return self.forward(z)


class Discriminator(nn.Module):
    """DCGAN Discriminator with spectral normalization and modern improvements.
    
    Args:
        image_size: Size of input images (assumes square)
        channels: Number of input channels
        hidden_dim: Hidden dimension size
        use_spectral_norm: Whether to use spectral normalization
    """
    
    def __init__(
        self,
        image_size: int = 28,
        channels: int = 1,
        hidden_dim: int = 64,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.use_spectral_norm = use_spectral_norm
        
        # Convolutional layers
        conv1 = nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False)
        conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
        conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False)
        conv4 = nn.Conv2d(hidden_dim * 4, 1, kernel_size=4, stride=1, padding=0, bias=False)
        
        # Apply spectral normalization if requested
        if use_spectral_norm:
            self.conv1 = spectral_norm(conv1)
            self.conv2 = spectral_norm(conv2)
            self.conv3 = spectral_norm(conv3)
            self.conv4 = spectral_norm(conv4)
        else:
            self.conv1 = conv1
            self.conv2 = conv2
            self.conv3 = conv3
            self.conv4 = conv4
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm2d(hidden_dim * 4)
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the discriminator.
        
        Args:
            x: Input images of shape (batch_size, channels, image_size, image_size)
            
        Returns:
            Discriminator output of shape (batch_size, 1)
        """
        # Apply convolutions with batch norm and LeakyReLU
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        
        return x.view(x.size(0), -1)


class DCGAN(nn.Module):
    """DCGAN model combining Generator and Discriminator.
    
    Args:
        generator_config: Configuration for the generator
        discriminator_config: Configuration for the discriminator
    """
    
    def __init__(
        self,
        generator_config: Dict[str, Any],
        discriminator_config: Dict[str, Any],
    ) -> None:
        super().__init__()
        
        self.generator = Generator(**generator_config)
        self.discriminator = Discriminator(**discriminator_config)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from noise.
        
        Args:
            z: Noise tensor
            
        Returns:
            Generated images
        """
        return self.generator(z)
    
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminate real vs fake images.
        
        Args:
            x: Input images
            
        Returns:
            Discriminator output
        """
        return self.discriminator(x)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample random images.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated images
        """
        return self.generator.sample(num_samples, device)
