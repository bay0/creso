"""
Core spectral layer implementing wave resonance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .logging import get_logger
from .validation import (
    validate_positive_int,
    validate_tensor_shape,
    validate_probability,
)
from .exceptions import ValidationError

logger = get_logger(__name__)


class WaveResonanceLayer(nn.Module):
    """Wave resonance layer with learnable spectral components.

    Implements learnable sinusoidal basis with optional Gaussian localization:
    z = sum_k [a_c_k * cos(omega_k · x + theta_k) + a_s_k * sin(omega_k · x + theta_k)] * envelope_k(x)

    Args:
        input_dim: Input dimension
        n_components: Number of spectral components
        localized: Whether to use Gaussian envelopes
        learn_centers: Whether to learn envelope centers (if localized=True)
        init_freq_scale: Scale for frequency initialization
    """

    def __init__(
        self,
        input_dim: int,
        n_components: int = 128,
        localized: bool = True,
        learn_centers: bool = True,
        init_freq_scale: float = 3.0,
    ):
        super().__init__()

        # Validate inputs
        validate_positive_int(input_dim, "input_dim")
        validate_positive_int(n_components, "n_components")

        self.input_dim = input_dim
        self.n_components = n_components
        self.localized = localized
        self.learn_centers = learn_centers
        self.init_freq_scale = init_freq_scale

        logger.debug(
            "Initializing WaveResonanceLayer",
            extra={
                "input_dim": input_dim,
                "n_components": n_components,
                "localized": localized,
                "learn_centers": learn_centers,
                "init_freq_scale": init_freq_scale,
            },
        )

        # Learnable frequency components: omega (n_components, input_dim)
        self.omega = nn.Parameter(
            torch.randn(n_components, input_dim) * init_freq_scale
        )

        # Phase shifts: theta (n_components,)
        self.theta = nn.Parameter(torch.randn(n_components) * 2 * math.pi)

        # Amplitudes for cos and sin components
        self.a_c = nn.Parameter(torch.randn(n_components) * 0.1)
        self.a_s = nn.Parameter(torch.randn(n_components) * 0.1)

        # Gaussian envelope parameters (if localized)
        if self.localized:
            if self.learn_centers:
                self.centers = nn.Parameter(torch.randn(n_components, input_dim))
            else:
                self.register_buffer("centers", torch.randn(n_components, input_dim))

            # Log-scale for numerical stability
            self.log_sigma = nn.Parameter(torch.zeros(n_components))

            logger.debug("Initialized Gaussian envelope parameters")

    def forward(
        self, x: torch.Tensor, freq_dropout_p: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, input_dim)
            freq_dropout_p: Frequency dropout probability

        Returns:
            z_spec: Spectral output (B, 1)
            C: Cosine components (B, n_components)
            S: Sine components (B, n_components)
            mask: Dropout mask (n_components,) if dropout applied, else None
        """
        # Skip validation during TorchScript tracing
        if not torch.jit.is_tracing():
            try:
                # Validate inputs
                validate_tensor_shape(x, (None, self.input_dim), "input tensor")
                validate_probability(freq_dropout_p, "freq_dropout_p")
            except (ValueError, TypeError, RuntimeError) as e:
                raise ValidationError(
                    f"Invalid input to WaveResonanceLayer: {e}"
                ) from e

        x.size(0)

        # Compute phase: omega · x + theta -> (B, K) - optimized with addmm
        phase = torch.addmm(self.theta.unsqueeze(0), x, self.omega.t())

        # Compute basis functions
        C = torch.cos(phase)  # (B, K)
        S = torch.sin(phase)  # (B, K)

        # Apply Gaussian envelopes if localized
        if self.localized:
            # Use device-appropriate epsilon to prevent numerical issues
            eps = torch.finfo(self.log_sigma.dtype).eps * 1000
            sigma = F.softplus(self.log_sigma) + eps  # (K,)

            # Memory-efficient distance computation: ||x - c_k||^2
            # Using broadcasting instead of cdist to avoid large intermediate matrices
            dist_sq = torch.sum(
                (x.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=2
            )  # (B, K)

            # Gaussian envelope: exp(-dist^2 / (2 * sigma^2))
            envelope = torch.exp(-dist_sq / (2 * sigma.unsqueeze(0) ** 2))  # (B, K)

            C = C * envelope
            S = S * envelope

        # Apply frequency dropout during training
        mask = None
        if self.training and freq_dropout_p > 0:
            # Ensure mask is on the same device as the output tensors C and S
            device = C.device
            mask = (
                torch.rand(self.n_components, device=device) > freq_dropout_p
            )  # (n_components,)
            C = C * mask.unsqueeze(0)
            S = S * mask.unsqueeze(0)

            active_components = torch.sum(mask).item()
            logger.debug(
                f"Frequency dropout: {active_components}/{self.n_components} components active"
            )

        # Combine with learnable amplitudes
        z_spec = torch.sum(self.a_c * C + self.a_s * S, dim=1, keepdim=True)  # (B, 1)

        return z_spec, C, S, mask

    def spectral_regularizers(
        self, l2_freq: float = 1e-4, group_l1: float = 1e-3, center_disp: float = 1e-5
    ) -> torch.Tensor:
        """Compute spectral regularization terms.

        Args:
            l2_freq: L2 penalty on frequencies
            group_l1: Group L1 penalty on amplitudes
            center_disp: Penalty on center dispersion

        Returns:
            Total regularization loss
        """
        reg_loss = 0.0

        # L2 regularization on frequencies
        if l2_freq > 0:
            reg_loss += l2_freq * torch.sum(self.omega**2)

        # Group L1 regularization on amplitudes (encourages sparsity)
        if group_l1 > 0:
            amp_norms = torch.sqrt(self.a_c**2 + self.a_s**2)
            reg_loss += group_l1 * torch.sum(amp_norms)

        # Center dispersion regularization (encourages diverse centers)
        if center_disp > 0 and self.localized and self.learn_centers:
            center_mean = torch.mean(self.centers, dim=0, keepdim=True)
            center_dist = torch.sum((self.centers - center_mean) ** 2)
            reg_loss += center_disp * center_dist

        return reg_loss

    def prune_by_amplitude(
        self, top_k: Optional[int] = None, threshold: Optional[float] = None
    ) -> None:
        """Prune low-amplitude spectral components.

        Args:
            top_k: Keep only top K components by amplitude
            threshold: Keep only components above threshold
        """
        with torch.no_grad():
            # Compute total amplitude per component
            total_amp = torch.sqrt(self.a_c**2 + self.a_s**2)

            if top_k is not None:
                # Keep top-k by amplitude
                _, indices = torch.topk(total_amp, min(top_k, self.n_components))
                logger.info(
                    f"Pruning to top {min(top_k, self.n_components)} components"
                )
                mask = torch.zeros_like(total_amp, dtype=torch.bool)
                mask[indices] = True
            elif threshold is not None:
                # Keep above threshold
                mask = total_amp >= threshold
                n_kept = torch.sum(mask).item()
                logger.info(
                    f"Pruning with threshold {threshold}: keeping {n_kept}/{self.n_components} components"
                )
            else:
                raise ValidationError("Must specify either top_k or threshold")

            # Zero out pruned components
            self.a_c.data[~mask] = 0
            self.a_s.data[~mask] = 0

    def get_frequency_magnitudes(self) -> torch.Tensor:
        """Get magnitude of each frequency component.

        Returns:
            Frequency magnitudes (n_components,)
        """
        return torch.sqrt(torch.sum(self.omega**2, dim=1))

    def get_amplitude_magnitudes(self) -> torch.Tensor:
        """Get amplitude magnitude of each component.

        Returns:
            Amplitude magnitudes (n_components,)
        """
        return torch.sqrt(self.a_c**2 + self.a_s**2)
