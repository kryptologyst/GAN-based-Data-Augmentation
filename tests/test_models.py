"""Unit tests for GAN models."""

import pytest
import torch
import torch.nn as nn
from src.models.dcgan import Generator, Discriminator, DCGAN
from src.models.losses import GANLoss, compute_gradient_penalty


class TestGenerator:
    """Test cases for Generator model."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = Generator(z_dim=100, hidden_dim=64, image_size=28, channels=1)
        
        assert generator.z_dim == 100
        assert generator.hidden_dim == 64
        assert generator.image_size == 28
        assert generator.channels == 1
    
    def test_generator_forward(self):
        """Test generator forward pass."""
        generator = Generator(z_dim=100, hidden_dim=64, image_size=28, channels=1)
        batch_size = 16
        z = torch.randn(batch_size, 100)
        
        output = generator(z)
        
        assert output.shape == (batch_size, 1, 28, 28)
        assert output.min() >= -1.0
        assert output.max() <= 1.0
    
    def test_generator_sample(self):
        """Test generator sampling."""
        generator = Generator(z_dim=100, hidden_dim=64, image_size=28, channels=1)
        device = torch.device("cpu")
        num_samples = 8
        
        samples = generator.sample(num_samples, device)
        
        assert samples.shape == (num_samples, 1, 28, 28)
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0


class TestDiscriminator:
    """Test cases for Discriminator model."""
    
    def test_discriminator_initialization(self):
        """Test discriminator initialization."""
        discriminator = Discriminator(
            image_size=28, channels=1, hidden_dim=64, use_spectral_norm=True
        )
        
        assert discriminator.image_size == 28
        assert discriminator.channels == 1
        assert discriminator.hidden_dim == 64
        assert discriminator.use_spectral_norm == True
    
    def test_discriminator_forward(self):
        """Test discriminator forward pass."""
        discriminator = Discriminator(
            image_size=28, channels=1, hidden_dim=64, use_spectral_norm=False
        )
        batch_size = 16
        x = torch.randn(batch_size, 1, 28, 28)
        
        output = discriminator(x)
        
        assert output.shape == (batch_size, 1)
    
    def test_discriminator_with_spectral_norm(self):
        """Test discriminator with spectral normalization."""
        discriminator = Discriminator(
            image_size=28, channels=1, hidden_dim=64, use_spectral_norm=True
        )
        batch_size = 16
        x = torch.randn(batch_size, 1, 28, 28)
        
        output = discriminator(x)
        
        assert output.shape == (batch_size, 1)


class TestDCGAN:
    """Test cases for DCGAN model."""
    
    def test_dcgan_initialization(self):
        """Test DCGAN initialization."""
        generator_config = {
            "z_dim": 100,
            "hidden_dim": 64,
            "image_size": 28,
            "channels": 1,
        }
        discriminator_config = {
            "image_size": 28,
            "channels": 1,
            "hidden_dim": 64,
        }
        
        model = DCGAN(generator_config, discriminator_config)
        
        assert isinstance(model.generator, Generator)
        assert isinstance(model.discriminator, Discriminator)
    
    def test_dcgan_forward(self):
        """Test DCGAN forward pass."""
        generator_config = {
            "z_dim": 100,
            "hidden_dim": 64,
            "image_size": 28,
            "channels": 1,
        }
        discriminator_config = {
            "image_size": 28,
            "channels": 1,
            "hidden_dim": 64,
        }
        
        model = DCGAN(generator_config, discriminator_config)
        batch_size = 16
        z = torch.randn(batch_size, 100)
        
        output = model(z)
        
        assert output.shape == (batch_size, 1, 28, 28)
    
    def test_dcgan_discriminate(self):
        """Test DCGAN discrimination."""
        generator_config = {
            "z_dim": 100,
            "hidden_dim": 64,
            "image_size": 28,
            "channels": 1,
        }
        discriminator_config = {
            "image_size": 28,
            "channels": 1,
            "hidden_dim": 64,
        }
        
        model = DCGAN(generator_config, discriminator_config)
        batch_size = 16
        x = torch.randn(batch_size, 1, 28, 28)
        
        output = model.discriminate(x)
        
        assert output.shape == (batch_size, 1)
    
    def test_dcgan_sample(self):
        """Test DCGAN sampling."""
        generator_config = {
            "z_dim": 100,
            "hidden_dim": 64,
            "image_size": 28,
            "channels": 1,
        }
        discriminator_config = {
            "image_size": 28,
            "channels": 1,
            "hidden_dim": 64,
        }
        
        model = DCGAN(generator_config, discriminator_config)
        device = torch.device("cpu")
        num_samples = 8
        
        samples = model.sample(num_samples, device)
        
        assert samples.shape == (num_samples, 1, 28, 28)


class TestGANLoss:
    """Test cases for GAN loss functions."""
    
    def test_gan_loss_vanilla(self):
        """Test vanilla GAN loss."""
        loss_fn = GANLoss(gan_mode="vanilla")
        
        # Test real prediction
        real_pred = torch.randn(16, 1)
        real_loss = loss_fn(real_pred, target_is_real=True)
        assert real_loss.item() >= 0
        
        # Test fake prediction
        fake_pred = torch.randn(16, 1)
        fake_loss = loss_fn(fake_pred, target_is_real=False)
        assert fake_loss.item() >= 0
    
    def test_gan_loss_lsgan(self):
        """Test LSGAN loss."""
        loss_fn = GANLoss(gan_mode="lsgan")
        
        # Test real prediction
        real_pred = torch.randn(16, 1)
        real_loss = loss_fn(real_pred, target_is_real=True)
        assert real_loss.item() >= 0
        
        # Test fake prediction
        fake_pred = torch.randn(16, 1)
        fake_loss = loss_fn(fake_pred, target_is_real=False)
        assert fake_loss.item() >= 0
    
    def test_gan_loss_wgangp(self):
        """Test WGAN-GP loss."""
        loss_fn = GANLoss(gan_mode="wgangp")
        
        # Test real prediction
        real_pred = torch.randn(16, 1)
        real_loss = loss_fn(real_pred, target_is_real=True)
        
        # Test fake prediction
        fake_pred = torch.randn(16, 1)
        fake_loss = loss_fn(fake_pred, target_is_real=False)
        
        # WGAN-GP losses can be negative
        assert isinstance(real_loss.item(), float)
        assert isinstance(fake_loss.item(), float)


class TestGradientPenalty:
    """Test cases for gradient penalty computation."""
    
    def test_gradient_penalty(self):
        """Test gradient penalty computation."""
        # Create a simple discriminator
        discriminator = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
        device = torch.device("cpu")
        batch_size = 16
        
        real_samples = torch.randn(batch_size, 784)
        fake_samples = torch.randn(batch_size, 784)
        
        gp = compute_gradient_penalty(
            discriminator, real_samples, fake_samples, device
        )
        
        assert gp.item() >= 0
        assert isinstance(gp.item(), float)


if __name__ == "__main__":
    pytest.main([__file__])
