#!/usr/bin/env python3
"""Setup script for the GAN data augmentation project."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> None:
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)


def main():
    """Main setup function."""
    print("🚀 Setting up GAN Data Augmentation Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create necessary directories
    directories = [
        "data",
        "checkpoints", 
        "logs",
        "assets/generated",
        "assets/evaluation",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Install dependencies
    run_command("pip install -r requirements.txt", "Installing dependencies")
    
    # Install pre-commit hooks
    run_command("pre-commit install", "Installing pre-commit hooks")
    
    # Run initial tests
    run_command("python -m pytest tests/ -v", "Running initial tests")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Train a model: python scripts/train.py")
    print("2. Generate samples: python scripts/sample.py --checkpoint checkpoints/best_model.ckpt")
    print("3. Launch demo: streamlit run demo/streamlit_app.py")
    print("4. Evaluate model: python scripts/evaluate.py --checkpoint checkpoints/best_model.ckpt")


if __name__ == "__main__":
    main()
