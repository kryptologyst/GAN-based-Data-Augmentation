"""Device utilities for automatic device selection."""

import torch
from typing import Union


def get_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """Get the best available device.
    
    Args:
        device: Device specification ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        PyTorch device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def get_device_info() -> dict:
    """Get information about the current device.
    
    Returns:
        Dictionary with device information
    """
    device = get_device()
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    
    if device.type == "cuda":
        info.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_current_device": torch.cuda.current_device(),
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_memory_allocated": torch.cuda.memory_allocated(),
            "cuda_memory_reserved": torch.cuda.memory_reserved(),
        })
    
    return info
