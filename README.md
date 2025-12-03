# GAN-based Data Augmentation

A production-ready implementation of DCGAN for data augmentation, featuring comprehensive evaluation metrics, interactive demos, and reproducible training pipelines.

## Overview

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) for generating synthetic images to augment training datasets. The implementation includes modern training techniques, comprehensive evaluation metrics, and interactive visualization tools.

## Features

- **Modern DCGAN Implementation**: Spectral normalization, proper weight initialization, and batch normalization
- **Comprehensive Evaluation**: FID, IS, Precision/Recall, and diversity metrics
- **Interactive Demo**: Streamlit-based web interface for sample generation and exploration
- **Reproducible Training**: Deterministic seeding, configuration management, and checkpointing
- **Production Ready**: Type hints, comprehensive documentation, and CI/CD pipeline

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/GAN-based-Data-Augmentation.git
cd GAN-based-Data-Augmentation

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom configuration
python scripts/train.py --config configs/custom_config.yaml

# Resume from checkpoint
python scripts/train.py --resume checkpoints/last.ckpt

# Fast development run
python scripts/train.py --fast-dev-run
```

### Sampling

```bash
# Generate samples
python scripts/sample.py --checkpoint checkpoints/best_model.ckpt --num-samples 64

# Generate interpolation
python scripts/sample.py --checkpoint checkpoints/best_model.ckpt --interpolation

# Generate latent traversal
python scripts/sample.py --checkpoint checkpoints/best_model.ckpt --traversal --dim 0
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/streamlit_app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/             # Model implementations
│   │   ├── dcgan.py        # DCGAN model
│   │   └── losses.py       # Loss functions
│   ├── data/               # Data handling
│   │   └── datasets.py     # Dataset classes
│   ├── training/           # Training utilities
│   │   ├── trainer.py      # PyTorch Lightning trainer
│   │   └── callbacks.py    # Custom callbacks
│   ├── evaluation/         # Evaluation metrics
│   │   └── metrics.py      # GAN evaluation metrics
│   └── utils/              # Utility functions
│       ├── device.py       # Device management
│       ├── seed.py         # Seeding utilities
│       └── visualization.py # Visualization tools
├── configs/                # Configuration files
│   ├── config.yaml         # Main configuration
│   ├── model/              # Model configurations
│   ├── data/               # Data configurations
│   └── training/           # Training configurations
├── scripts/                # Training and sampling scripts
│   ├── train.py            # Training script
│   └── sample.py           # Sampling script
├── demo/                   # Interactive demos
│   └── streamlit_app.py    # Streamlit demo
├── tests/                  # Unit tests
├── assets/                 # Generated samples and visualizations
├── checkpoints/            # Model checkpoints
└── logs/                   # Training logs
```

## Configuration

The project uses OmegaConf for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration with experiment settings
- `configs/model/dcgan.yaml`: Model architecture and training parameters
- `configs/data/mnist.yaml`: Dataset and data loading settings
- `configs/training/basic.yaml`: Training hyperparameters and callbacks

### Example Configuration

```yaml
# configs/config.yaml
experiment_name: "data_augmentation_gan"
seed: 42
device: "auto"

model:
  generator:
    z_dim: 100
    hidden_dim: 64
    image_size: 28
    channels: 1
  
  discriminator:
    image_size: 28
    channels: 1
    hidden_dim: 64

training:
  max_epochs: 100
  batch_size: 64
  learning_rate: 0.0002
```

## Model Architecture

### Generator
- Input: 100-dimensional noise vector
- Architecture: Transposed convolutions with batch normalization
- Output: 28x28 grayscale images
- Activation: ReLU (hidden), Tanh (output)

### Discriminator
- Input: 28x28 grayscale images
- Architecture: Convolutional layers with spectral normalization
- Output: Single scalar (real/fake probability)
- Activation: LeakyReLU (hidden), Sigmoid (output)

## Training Features

- **Spectral Normalization**: Stabilizes discriminator training
- **Gradient Penalty**: For WGAN-GP loss variant
- **Mixed Precision**: Automatic mixed precision training
- **Learning Rate Scheduling**: Cosine annealing scheduler
- **Early Stopping**: Prevents overfitting
- **Model Checkpointing**: Saves best and last models

## Evaluation Metrics

- **Fréchet Inception Distance (FID)**: Measures quality and diversity
- **Inception Score (IS)**: Measures quality and diversity
- **Precision and Recall**: Measures quality and coverage
- **LPIPS Diversity**: Measures perceptual diversity

## Data Augmentation Pipeline

The trained generator can be used for data augmentation:

```python
from src.data.datasets import AugmentedDataset

# Create augmented dataset
augmented_dataset = AugmentedDataset(
    real_dataset=train_dataset,
    generator=trained_generator,
    device=device,
    augmentation_ratio=0.5,  # 50% generated samples
)

# Use in training
augmented_loader = DataLoader(augmented_dataset, batch_size=64)
```

## Interactive Demo

The Streamlit demo provides:

- **Sample Generation**: Generate random samples with controllable seed
- **Latent Interpolation**: Interpolate between two noise vectors
- **Latent Traversal**: Traverse along specific latent dimensions
- **Real-time Visualization**: Interactive plots and image grids

## Development

### Code Quality

```bash
# Format code
black src/ scripts/ demo/

# Lint code
ruff check src/ scripts/ demo/

# Run tests
pytest tests/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Model Card

### Intended Use
- Data augmentation for image classification tasks
- Research and education on generative models
- Understanding latent space representations

### Training Data
- MNIST handwritten digits dataset
- 60,000 training images, 10,000 test images
- 28x28 grayscale images

### Performance
- FID: ~15-25 (lower is better)
- IS: ~2.5-3.5 (higher is better)
- Training time: ~2-4 hours on GPU

### Limitations
- Trained on MNIST digits only
- May not generalize to other image types
- Generated images may lack fine details

### Bias and Fairness
- No demographic bias (digits are neutral)
- May inherit biases from training data distribution
- Generated samples should be validated before use

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{gan_data_augmentation,
  title={GAN-based Data Augmentation},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/GAN-based-Data-Augmentation}
}
```

## Acknowledgments

- PyTorch Lightning team for the training framework
- Clean-FID authors for evaluation metrics
- Streamlit team for the demo framework
- MNIST dataset creators
# GAN-based-Data-Augmentation
