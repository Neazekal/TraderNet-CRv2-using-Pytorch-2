"""
Multi-GPU and Distributed Training Utilities.

Provides utilities for:
- Detecting available GPUs
- Wrapping models with DataParallel for multi-GPU training
- Device management and placement

Usage:
    from utils.distributed import get_device, setup_multi_gpu, wrap_model_multi_gpu
    
    device, gpu_ids = get_device()
    model = wrap_model_multi_gpu(model, gpu_ids)
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Union


def get_available_gpus() -> List[int]:
    """
    Get list of available GPU device IDs.
    
    Returns:
        List of GPU IDs (e.g., [0, 1, 2] for 3 GPUs)
    """
    if not torch.cuda.is_available():
        return []
    
    return list(range(torch.cuda.device_count()))


def get_device(prefer_multi_gpu: bool = True) -> Tuple[torch.device, List[int]]:
    """
    Get the best available device and list of GPU IDs.
    
    Args:
        prefer_multi_gpu: If True, return all available GPUs for DataParallel
        
    Returns:
        Tuple of (primary device, list of GPU IDs)
        - If no GPU: (cpu, [])
        - If single GPU: (cuda:0, [0])
        - If multi GPU: (cuda:0, [0, 1, ...])
    """
    gpu_ids = get_available_gpus()
    
    if not gpu_ids:
        return torch.device('cpu'), []
    
    if not prefer_multi_gpu:
        return torch.device('cuda:0'), [0]
    
    return torch.device('cuda:0'), gpu_ids


def wrap_model_multi_gpu(
    model: nn.Module,
    gpu_ids: List[int],
    output_device: Optional[int] = None
) -> nn.Module:
    """
    Wrap a model with DataParallel for multi-GPU training.
    
    Args:
        model: PyTorch model to wrap
        gpu_ids: List of GPU IDs to use
        output_device: GPU ID for gathering outputs (default: gpu_ids[0])
        
    Returns:
        Model wrapped with DataParallel if multiple GPUs, otherwise original model
    """
    if len(gpu_ids) <= 1:
        return model
    
    if output_device is None:
        output_device = gpu_ids[0]
    
    print(f"Wrapping model with DataParallel on GPUs: {gpu_ids}")
    return nn.DataParallel(model, device_ids=gpu_ids, output_device=output_device)


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Unwrap a DataParallel model to get the underlying module.
    
    Useful for saving checkpoints or accessing model attributes.
    
    Args:
        model: Model that may be wrapped with DataParallel
        
    Returns:
        Underlying model without DataParallel wrapper
    """
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def get_model_device(model: nn.Module) -> torch.device:
    """
    Get the device of a model's parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Device of the model's first parameter
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def print_gpu_info():
    """Print information about available GPUs."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"\nGPU Information:")
    print(f"  CUDA available: True")
    print(f"  Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024 ** 3)
        print(f"  GPU {i}: {props.name}")
        print(f"    - Memory: {memory_gb:.1f} GB")
        print(f"    - Compute capability: {props.major}.{props.minor}")
    
    if num_gpus > 1:
        print(f"\n  Multi-GPU training: ENABLED ({num_gpus} GPUs)")
    else:
        print(f"\n  Multi-GPU training: DISABLED (single GPU)")


class MultiGPUManager:
    """
    Manager class for multi-GPU training setup.
    
    Handles device detection, model wrapping, and provides utilities
    for checkpoint saving/loading with DataParallel models.
    
    Example:
        manager = MultiGPUManager()
        model = manager.wrap(model)
        
        # For saving checkpoint
        state_dict = manager.get_state_dict(model)
        
        # For loading checkpoint
        manager.load_state_dict(model, state_dict)
    """
    
    def __init__(self, prefer_multi_gpu: bool = True, verbose: bool = True):
        """
        Initialize multi-GPU manager.
        
        Args:
            prefer_multi_gpu: Whether to use all available GPUs
            verbose: Print GPU information
        """
        self.device, self.gpu_ids = get_device(prefer_multi_gpu)
        self.is_multi_gpu = len(self.gpu_ids) > 1
        
        if verbose:
            print_gpu_info()
            print(f"\nUsing device: {self.device}")
            if self.is_multi_gpu:
                print(f"Multi-GPU mode: Using GPUs {self.gpu_ids}")
    
    def wrap(self, model: nn.Module) -> nn.Module:
        """Wrap model for multi-GPU if available."""
        model = model.to(self.device)
        if self.is_multi_gpu:
            model = wrap_model_multi_gpu(model, self.gpu_ids)
        return model
    
    def unwrap(self, model: nn.Module) -> nn.Module:
        """Unwrap DataParallel model."""
        return unwrap_model(model)
    
    def get_state_dict(self, model: nn.Module) -> dict:
        """Get state dict, handling DataParallel wrapper."""
        return unwrap_model(model).state_dict()
    
    def load_state_dict(self, model: nn.Module, state_dict: dict):
        """Load state dict, handling DataParallel wrapper."""
        unwrap_model(model).load_state_dict(state_dict)
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the primary device."""
        return tensor.to(self.device)


if __name__ == '__main__':
    """Test multi-GPU utilities."""
    print("Testing Multi-GPU Utilities")
    print("=" * 50)
    
    # Print GPU info
    print_gpu_info()
    
    # Test device detection
    device, gpu_ids = get_device()
    print(f"\nDetected device: {device}")
    print(f"GPU IDs: {gpu_ids}")
    
    # Test with a simple model
    model = nn.Linear(10, 5)
    model = model.to(device)
    
    if len(gpu_ids) > 1:
        wrapped = wrap_model_multi_gpu(model, gpu_ids)
        print(f"Model wrapped: {type(wrapped)}")
        
        unwrapped = unwrap_model(wrapped)
        print(f"Model unwrapped: {type(unwrapped)}")
    
    # Test manager
    print("\nTesting MultiGPUManager:")
    manager = MultiGPUManager(verbose=False)
    print(f"  Device: {manager.device}")
    print(f"  Multi-GPU: {manager.is_multi_gpu}")
    
    print("\n✓ Multi-GPU utilities test passed!")
