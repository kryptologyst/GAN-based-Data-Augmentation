"""Seeding utilities for reproducible experiments."""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variables for additional determinism
        import os
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_random_state() -> dict:
    """Get current random state for all libraries.
    
    Returns:
        Dictionary containing random states
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def set_random_state(state: dict) -> None:
    """Set random state from saved state.
    
    Args:
        state: Dictionary containing random states
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"])
