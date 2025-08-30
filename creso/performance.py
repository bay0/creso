"""
Performance optimization utilities for CReSO.

Provides optimized configurations and utilities for faster training and inference.
"""

import logging
import os
from typing import Optional

from .config import (
    CReSOConfiguration, ModelArchitectureConfig, TrainingConfig, 
    SystemConfig, FrequencySeedingConfig
)

def create_fast_config(
    input_dim: int,
    n_samples: Optional[int] = None,
    n_classes: int = 2,
    device: str = "cpu"
) -> CReSOConfiguration:
    """
    Create a fast configuration optimized for speed.
    
    Args:
        input_dim: Number of input features
        n_samples: Number of training samples (for batch size optimization)
        n_classes: Number of classes (affects component sizing)
        device: Device to use
        
    Returns:
        Optimized configuration for speed
    """
    # Adaptive component sizing based on problem complexity
    if n_classes == 2:
        base_components = min(32, max(8, input_dim // 2))
    else:
        # Multiclass needs more components
        base_components = min(64, max(16, input_dim))
    
    # Adaptive batch size
    if n_samples is not None:
        batch_size = min(256, max(32, n_samples // 8))
    else:
        batch_size = 128
    
    # Reduced epochs for speed
    epochs = 10 if n_samples and n_samples < 1000 else 15
    
    arch_config = ModelArchitectureConfig(
        input_dim=input_dim,
        n_components=base_components,
        geometric_hidden_dim=min(64, max(16, input_dim // 2))
    )
    
    train_config = TrainingConfig(
        max_epochs=epochs,
        learning_rate=0.005,  # Slightly higher for faster convergence
        batch_size=batch_size,
        early_stopping_patience=5,  # Shorter patience
        use_automatic_mixed_precision=device.startswith("cuda")  # Only use AMP on GPU
    )
    
    # Faster frequency seeding
    seed_config = FrequencySeedingConfig(
        enable_frequency_seeding=True,
        seeding_fraction=0.3,  # More seeding for stability
        max_autocorr_lag=min(50, max(10, input_dim))  # Adaptive lag
    )
    
    system_config = SystemConfig(
        device=device,
        pin_memory=device.startswith("cuda"),
        num_workers=0 if n_samples and n_samples < 1000 else 2
    )
    
    return CReSOConfiguration(
        architecture=arch_config,
        training=train_config,
        frequency_seeding=seed_config,
        system=system_config,
        name="fast"
    )

def enable_performance_mode():
    """Enable performance optimizations globally."""
    # Reduce logging verbosity
    logging.getLogger('creso').setLevel(logging.WARNING)
    
    # Set environment variables for optimization
    os.environ['TORCH_CUDNN_BENCHMARK'] = '1'
    os.environ['OMP_NUM_THREADS'] = str(min(4, os.cpu_count() or 1))
    
def disable_performance_mode():
    """Disable performance mode and restore normal logging."""
    logging.getLogger('creso').setLevel(logging.INFO)

def get_optimal_batch_size(n_samples: int, input_dim: int, device: str = "cpu") -> int:
    """Calculate optimal batch size based on dataset characteristics."""
    if device.startswith("cuda"):
        # GPU memory considerations
        if input_dim <= 20:
            base_size = 512
        elif input_dim <= 100:
            base_size = 256
        else:
            base_size = 128
    else:
        # CPU optimization
        base_size = 128
    
    # Adjust for dataset size
    optimal_size = min(base_size, max(32, n_samples // 10))
    return optimal_size

def estimate_training_time(
    n_samples: int, 
    input_dim: int, 
    n_components: int,
    epochs: int,
    device: str = "cpu"
) -> float:
    """
    Estimate training time in seconds.
    
    Based on empirical performance measurements.
    """
    # Base time per sample per epoch (seconds)
    if device.startswith("cuda"):
        base_time = 2e-5  # GPU is ~10x faster
    else:
        base_time = 2e-4  # CPU baseline
    
    # Complexity factors
    complexity_factor = (n_components / 32) * (input_dim / 20)
    batch_factor = 0.8  # Batching provides ~20% speedup
    
    total_time = (
        n_samples * epochs * base_time * 
        complexity_factor * batch_factor
    )
    
    return total_time