"""
Utility functions for CReSO.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Union

from .logging import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set global random seed for reproducibility.

    Args:
        seed: Random seed
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def as_tensor(
    x: Union[np.ndarray, torch.Tensor, list],
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert input to torch tensor with specified device and dtype.

    Args:
        x: Input data
        device: Target device
        dtype: Target dtype

    Returns:
        Torch tensor
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype)
    else:
        x = x.to(dtype=dtype)

    if device is not None:
        x = x.to(device)

    return x


class Standardizer(nn.Module):
    """Data standardization module with state persistence.

    Computes and applies z-score normalization: (x - mean) / std
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", None)
        self.register_buffer("std", None)
        self.fitted = False

    def fit(self, X: torch.Tensor) -> "Standardizer":
        """Fit standardizer to data.

        Args:
            X: Input data of shape (N, D)

        Returns:
            Self for chaining
        """
        if X.dim() != 2:
            raise ValueError(f"Expected 2D input, got {X.dim()}D")

        mean = X.mean(dim=0, keepdim=False)
        std = X.std(dim=0, keepdim=False, unbiased=False)

        # Avoid division by zero
        std = torch.clamp(std, min=1e-6)

        # Properly set buffers
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.fitted = True
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply standardization.

        Args:
            X: Input data of shape (N, D)

        Returns:
            Standardized data
        """
        if not self.fitted:
            raise RuntimeError("Standardizer must be fitted before transform")

        if self.mean is None or self.std is None:
            raise RuntimeError(
                f"Standardizer buffers not properly loaded. fitted={self.fitted}, mean={self.mean is not None}, std={self.std is not None}"
            )

        # Validate tensor contents for NaN/Inf values
        if isinstance(self.mean, torch.Tensor):
            if torch.isnan(self.mean).any() or torch.isinf(self.mean).any():
                raise RuntimeError(
                    "Standardizer mean buffer contains NaN or Inf values"
                )
        if isinstance(self.std, torch.Tensor):
            if torch.isnan(self.std).any() or torch.isinf(self.std).any():
                raise RuntimeError("Standardizer std buffer contains NaN or Inf values")
            if (self.std <= 0).any():
                raise RuntimeError(
                    "Standardizer std buffer contains non-positive values"
                )

        # Buffers are always tensors after fitting, no need for conditional creation
        return (X - self.mean) / self.std

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step.

        Args:
            X: Input data of shape (N, D)

        Returns:
            Standardized data
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Reverse standardization.

        Args:
            X: Standardized data

        Returns:
            Original scale data
        """
        if not self.fitted:
            raise RuntimeError("Standardizer must be fitted before inverse_transform")

        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer buffers not properly loaded")
        mean_tensor = torch.as_tensor(self.mean)
        std_tensor = torch.as_tensor(self.std)
        return X * std_tensor + mean_tensor

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Get state dict including fit status."""
        state = super().state_dict(*args, **kwargs)
        state["fitted"] = self.fitted
        return state

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Load state dict including fit status."""
        state_dict = state_dict.copy()  # Don't modify original
        self.fitted = state_dict.pop("fitted", False)

        # Manually set buffers since they may not be properly handled by parent class
        if "mean" in state_dict:
            self.register_buffer("mean", state_dict["mean"])
        if "std" in state_dict:
            self.register_buffer("std", state_dict["std"])

        # Load any remaining parameters
        remaining_state = {
            k: v for k, v in state_dict.items() if k not in ["mean", "std"]
        }
        if remaining_state:
            super().load_state_dict(remaining_state, strict=False)


def init_constant_q_frequencies(
    omega: torch.Tensor,
    q_factor: float = 1.0,
    min_freq: float = 0.1,
    max_freq: float = 10.0,
) -> None:
    """Initialize frequencies with constant Q-factor relationship.

    Ensures frequencies are distributed to maintain proper bandwidth scaling.

    Args:
        omega: Frequency parameter tensor (n_components, input_dim)
        q_factor: Target Q-factor (bandwidth/frequency ratio)
        min_freq: Minimum frequency norm
        max_freq: Maximum frequency norm
    """
    with torch.no_grad():
        n_components, input_dim = omega.shape

        # Generate frequency norms in logarithmic scale for better coverage
        freq_norms = torch.logspace(
            np.log10(min_freq),
            np.log10(max_freq),
            steps=n_components,
            device=omega.device,
        )

        # Generate random directions and scale to desired norms
        directions = torch.randn_like(omega)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        # Set frequencies with proper norms
        omega.copy_(directions * freq_norms.unsqueeze(-1))


def init_phase_diversity(phase: torch.Tensor, diversity_factor: float = 1.0) -> None:
    """Initialize phases for maximum diversity to enable interference.

    Args:
        phase: Phase parameter tensor (n_components,)
        diversity_factor: Factor controlling phase diversity (0=uniform, 1=maximum diversity)
    """
    with torch.no_grad():
        n_components = phase.shape[0]

        if diversity_factor > 0.5:
            # Maximum diversity: evenly spaced phases
            phases = torch.linspace(0, 2 * np.pi, n_components + 1)[:-1]
            # Add small random perturbation
            phases += diversity_factor * 0.2 * torch.randn(n_components)
        else:
            # Random phases with controlled variance
            phases = torch.randn(n_components) * 2 * np.pi * diversity_factor

        phase.copy_(phases.to(phase.device))


def init_envelope_centers_kmeans(
    centers: torch.Tensor, X_sample: torch.Tensor, n_iter: int = 10
) -> None:
    """Initialize envelope centers using k-means++ style initialization.

    Args:
        centers: Center parameter tensor (n_components, input_dim)
        X_sample: Sample data for initialization (N, input_dim)
        n_iter: Number of k-means iterations
    """
    with torch.no_grad():
        n_components, input_dim = centers.shape
        n_samples = X_sample.shape[0]
        device = centers.device

        if n_samples < n_components:
            # Not enough samples, use random initialization
            centers.normal_(0, 1)
            return

        # k-means++ initialization
        # First center: random sample
        indices = torch.randperm(n_samples, device=device)
        centers[0] = X_sample[indices[0]]

        # Subsequent centers: probability proportional to distance
        for i in range(1, n_components):
            # Compute distances to nearest existing center
            dists = torch.cdist(X_sample, centers[:i])  # (N, i)
            min_dists = torch.min(dists, dim=-1)[0]  # (N,)

            # Sample proportional to squared distance
            probs = min_dists**2
            probs = probs / torch.sum(probs)

            # Sample new center
            idx = torch.multinomial(probs, 1).item()
            centers[i] = X_sample[idx]

        # Optional: run a few k-means iterations to refine
        for _ in range(n_iter):
            # Assign points to nearest centers
            dists = torch.cdist(X_sample, centers)  # (N, K)
            assignments = torch.argmin(dists, dim=-1)  # (N,)

            # Update centers
            for k in range(n_components):
                mask = assignments == k
                if torch.sum(mask) > 0:
                    centers[k] = torch.mean(X_sample[mask], dim=0)


def apply_wave_physics_initialization(
    wave_model,
    X_sample: torch.Tensor,
    q_factor: float = 1.0,
    phase_diversity: float = 1.0,
    use_kmeans_centers: bool = True,
) -> None:
    """Apply comprehensive wave physics initialization to a wave model.

    Args:
        wave_model: CReSOWaveModel instance
        X_sample: Sample data for initialization (N, input_dim)
        q_factor: Q-factor for frequency initialization
        phase_diversity: Phase diversity factor
        use_kmeans_centers: Whether to use k-means for center initialization
    """
    if not hasattr(wave_model, "wave_packets"):
        return  # Not a wave model

    wave_packets = wave_model.wave_packets

    # Initialize frequencies with constant-Q relationship
    init_constant_q_frequencies(wave_packets.omega, q_factor=q_factor)

    # Initialize phases for diversity
    init_phase_diversity(wave_packets.phase, diversity_factor=phase_diversity)

    # Initialize envelope centers
    if use_kmeans_centers and hasattr(wave_packets, "centers"):
        init_envelope_centers_kmeans(wave_packets.centers, X_sample)

    logger.info("Applied wave physics initialization")
    logger.info(f"  Q-factor: {q_factor}")
    logger.info(f"  Phase diversity: {phase_diversity}")
    logger.info(f"  K-means centers: {use_kmeans_centers}")

    # Log frequency statistics
    freq_norms = torch.norm(wave_packets.omega, dim=-1)
    logger.info(f"  Frequency range: [{freq_norms.min():.3f}, {freq_norms.max():.3f}]")
    logger.info(f"  Mean frequency: {freq_norms.mean():.3f}")


def safe_makedirs(path: str, exist_ok: bool = True) -> None:
    """Safely create directories with path validation.

    Args:
        path: Directory path to create
        exist_ok: Don't raise error if directory exists

    Raises:
        ValueError: If path is unsafe (contains traversal patterns)
    """
    import os
    import pathlib

    # Normalize and resolve the path
    try:
        normalized_path = os.path.normpath(path)
        resolved_path = str(pathlib.Path(normalized_path).resolve())
    except Exception as e:
        raise ValueError(f"Invalid path: {path}") from e

    # Check for path traversal attempts and reject unsafe paths
    current_dir = os.path.abspath(os.getcwd())
    absolute_path = os.path.abspath(resolved_path)

    if ".." in normalized_path:
        raise ValueError(f"Path traversal attempt detected in path: {path}")

    # For absolute paths, allow temp directories and current working directory
    if normalized_path.startswith("/"):
        import tempfile

        temp_dir = tempfile.gettempdir()
        allowed_prefixes = [current_dir, temp_dir]
        if not any(absolute_path.startswith(prefix) for prefix in allowed_prefixes):
            logger.warning(f"Path outside standard directories: {path}")
            # Allow but warn for non-standard paths to maintain compatibility

    # Create directory
    os.makedirs(resolved_path, exist_ok=exist_ok)
