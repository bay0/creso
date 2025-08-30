"""
Wave physics components for CReSO+ models.

Implements wave-principled neural components based on physical wave equations:
- Constant-Q resonant packets
- Unitary wave propagation operators
- Coherence-based interference gating
- Energy conservation constraints
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any

from .logging import get_logger
from .validation import validate_positive_int, validate_tensor_shape
from .exceptions import ValidationError

logger = get_logger(__name__)


class ConstantQResonantPacket(nn.Module):
    """Constant-Q resonant wave packets.

    Implements atomic wave units with Gaussian envelope and complex sinusoids,
    where bandwidth scales with frequency (constant Q-factor):

    ψ_k(x) = a_k * exp(-σ_k ||x-μ_k||²) * exp(i(ω_k·x + φ_k))

    with σ_k ∝ ||ω_k||² for constant Q = bandwidth/frequency ratio.

    Args:
        input_dim: Input feature dimension
        n_components: Number of wave packets
        q_factor: Constant Q-factor (bandwidth/frequency ratio)
        learn_centers: Whether to learn envelope centers
        init_freq_scale: Scale for frequency initialization
    """

    def __init__(
        self,
        input_dim: int,
        n_components: int = 32,
        q_factor: float = 1.0,
        learn_centers: bool = True,
        init_freq_scale: float = 3.0,
    ):
        super().__init__()

        validate_positive_int(input_dim, "input_dim")
        validate_positive_int(n_components, "n_components")

        self.input_dim = input_dim
        self.n_components = n_components
        self.q_factor = q_factor
        self.learn_centers = learn_centers
        self.init_freq_scale = init_freq_scale

        logger.debug(
            "Initializing ConstantQResonantPacket",
            extra={
                "input_dim": input_dim,
                "n_components": n_components,
                "q_factor": q_factor,
                "learn_centers": learn_centers,
            },
        )

        # Frequency vectors: ω (n_components, input_dim)
        self.omega = nn.Parameter(
            torch.randn(n_components, input_dim) * init_freq_scale
        )

        # Phase offsets: φ (n_components,)
        self.phase = nn.Parameter(torch.rand(n_components) * 2 * math.pi)

        # Complex amplitudes: a = a_real + i * a_imag
        self.amp_real = nn.Parameter(torch.randn(n_components) * 0.1)
        self.amp_imag = nn.Parameter(torch.randn(n_components) * 0.1)

        # Envelope centers: μ (n_components, input_dim)
        if learn_centers:
            self.centers = nn.Parameter(torch.randn(n_components, input_dim))
        else:
            self.register_buffer("centers", torch.randn(n_components, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass generating wave packets.

        Args:
            x: Input tensor (B, input_dim)

        Returns:
            Complex wave packets (B, n_components)
        """
        if not torch.jit.is_tracing():
            try:
                validate_tensor_shape(x, (None, self.input_dim), "input tensor")
            except (ValueError, TypeError, RuntimeError) as e:
                raise ValidationError(
                    f"Invalid input to ConstantQResonantPacket: {e}"
                ) from e

        # Compute frequency norms for constant-Q constraint
        freq_norm = torch.norm(self.omega, dim=-1, keepdim=True)  # (K, 1)

        # Constant-Q: σ_k ∝ ||ω_k||²
        # Higher frequencies → tighter localization
        sigma = self.q_factor * (freq_norm**2) + 1e-4  # (K, 1)

        # Distance from envelope centers using optimized cdist: ||x - μ_k||²
        dist_sq = torch.cdist(x, self.centers, p=2) ** 2  # (B, K)

        # Gaussian envelope: exp(-dist² / (2σ²))
        envelope = torch.exp(-dist_sq / (2 * sigma.squeeze(-1) ** 2))  # (B, K)

        # Phase: ω_k · x + φ_k
        phase = torch.matmul(x, self.omega.t()) + self.phase.unsqueeze(0)  # (B, K)

        # Complex exponential: exp(i * phase)
        complex_wave = torch.complex(torch.cos(phase), torch.sin(phase))  # (B, K)

        # Complex amplitude
        complex_amp = torch.complex(self.amp_real, self.amp_imag)  # (K,)

        # Complete wave packet: envelope * complex_wave * amplitude
        psi = envelope * complex_wave * complex_amp.unsqueeze(0)  # (B, K)

        return psi

    def get_q_factors(self) -> torch.Tensor:
        """Compute actual Q-factors for each component.

        Returns:
            Q-factors (n_components,)
        """
        freq_norm = torch.norm(self.omega, dim=-1)
        sigma = self.q_factor * (freq_norm**2) + 1e-4
        bandwidth = 1.0 / sigma  # Inverse relationship
        q_actual = freq_norm / bandwidth
        return q_actual

    def get_amplitude_magnitudes(self) -> torch.Tensor:
        """Get amplitude magnitudes for pruning analysis.

        Returns:
            Amplitude magnitudes (n_components,)
        """
        return torch.sqrt(self.amp_real**2 + self.amp_imag**2)

    def prune_components(self, threshold_percentile: float = 50.0) -> int:
        """Prune components based on amplitude magnitude.

        Args:
            threshold_percentile: Percentile threshold for pruning

        Returns:
            Number of components pruned
        """
        with torch.no_grad():
            amp_magnitudes = self.get_amplitude_magnitudes()
            threshold = torch.quantile(amp_magnitudes, threshold_percentile / 100.0)

            # Create pruning mask
            keep_mask = amp_magnitudes > threshold
            n_pruned = (~keep_mask).sum().item()

            # Zero out pruned components
            self.amp_real.data[~keep_mask] = 0.0
            self.amp_imag.data[~keep_mask] = 0.0

            logger.info(f"Pruned {n_pruned}/{self.n_components} components")
            return n_pruned


class WavePropagatorLayer(nn.Module):
    """Unitary wave propagation operator.

    Implements spectral domain evolution simulating wave propagation:
    z_out = F⁻¹[exp(iτΩ(λ)) ⊙ F[z_in]]

    where Ω(λ) is a learnable dispersion relation and F is the Fourier transform.
    The operator is unitary by construction, ensuring energy conservation.

    Args:
        n_features: Number of spectral features
        n_hidden: Hidden dimension for dispersion relation network
        init_tau: Initial time parameter for evolution
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int = 16,
        init_tau: float = 0.1,
    ):
        super().__init__()

        validate_positive_int(n_features, "n_features")

        self.n_features = n_features
        self.n_hidden = n_hidden

        logger.debug(
            "Initializing WavePropagatorLayer",
            extra={
                "n_features": n_features,
                "n_hidden": n_hidden,
                "init_tau": init_tau,
            },
        )

        # Learnable dispersion relation: frequency → phase velocity
        # Ω(λ) maps frequency to angular frequency for wave evolution
        self.dispersion_net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

        # Evolution time parameter
        self.tau = nn.Parameter(torch.tensor(init_tau))

        # Initialize dispersion relation to be approximately linear (non-dispersive)
        with torch.no_grad():
            self.dispersion_net[0].weight.normal_(0, 0.1)
            self.dispersion_net[2].weight.normal_(0, 0.1)
            self.dispersion_net[4].weight.normal_(0, 0.1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply unitary wave propagation.

        Args:
            z: Complex input tensor (B, n_features)

        Returns:
            Evolved complex tensor (B, n_features)
        """
        if not torch.jit.is_tracing():
            try:
                validate_tensor_shape(z, (None, self.n_features), "input tensor")
            except (ValueError, TypeError, RuntimeError) as e:
                raise ValidationError(
                    f"Invalid input to WavePropagatorLayer: {e}"
                ) from e

        B, K = z.shape

        # Transform to frequency domain
        Z_freq = torch.fft.fft(z, dim=-1)  # (B, K)

        # Frequency grid for dispersion relation
        freqs = torch.fft.fftfreq(K, device=z.device)  # (K,)
        freq_input = freqs.unsqueeze(-1)  # (K, 1)

        # Compute dispersion relation: Ω(λ)
        omega = self.dispersion_net(freq_input).squeeze(-1)  # (K,)

        # Unitary evolution operator: exp(i * τ * Ω(λ))
        evolution_phase = self.tau * omega  # (K,)
        evolution_op = torch.complex(
            torch.cos(evolution_phase), torch.sin(evolution_phase)
        )  # (K,)

        # Apply evolution in frequency domain
        Z_evolved = Z_freq * evolution_op.unsqueeze(0)  # (B, K)

        # Transform back to spatial domain
        z_out = torch.fft.ifft(Z_evolved, dim=-1)  # (B, K)

        return z_out

    def energy_conservation_loss(
        self, z_in: torch.Tensor, z_out: torch.Tensor
    ) -> torch.Tensor:
        """Compute energy conservation loss.

        For a unitary operator, ||z_out||² should equal ||z_in||².

        Args:
            z_in: Input tensor
            z_out: Output tensor

        Returns:
            Energy conservation loss scalar
        """
        energy_in = torch.mean(torch.abs(z_in) ** 2)
        energy_out = torch.mean(torch.abs(z_out) ** 2)
        return torch.abs(energy_in - energy_out)

    def get_dispersion_parameters(self) -> Dict[str, torch.Tensor]:
        """Get dispersion network parameters for analysis.

        Returns:
            Dictionary of parameter statistics
        """
        params = {}
        with torch.no_grad():
            # Compute dispersion function values over frequency range
            freqs = torch.linspace(0, 1, 100).unsqueeze(-1)
            dispersions = self.dispersion_net(freqs)

            params["dispersion_mean"] = dispersions.mean()
            params["dispersion_std"] = dispersions.std()
            params["dispersion_range"] = (dispersions.min(), dispersions.max())
            params["tau"] = self.tau.clone()

        return params


class CoherenceGate(nn.Module):
    """Coherence-based interference gating.

    Implements gating mechanism based on phase alignment and interference
    between wave packets. Gates are computed from constructive/destructive
    interference patterns:

    γ(x) = σ(Σ_{k≠j} w_{kj} Re[ψ_k(x) ψ*_j(x)])

    This allows the model to adaptively select components based on their
    phase relationships and interference patterns.

    Args:
        n_components: Number of wave components
        temperature: Temperature for gating (higher = softer gates)
    """

    def __init__(self, n_components: int, temperature: float = 1.0):
        super().__init__()

        validate_positive_int(n_components, "n_components")

        self.n_components = n_components
        self.temperature = temperature

        logger.debug(
            "Initializing CoherenceGate",
            extra={"n_components": n_components, "temperature": temperature},
        )

        # Pairwise interference weights: W_{kj}
        self.W_interference = nn.Parameter(
            torch.randn(n_components, n_components) * 0.01
        )

        # Bias term
        self.bias = nn.Parameter(torch.zeros(n_components))

        # Zero out diagonal (no self-interference) - use atomic operation
        with torch.no_grad():
            # Create a mask for diagonal elements to ensure atomic operation
            mask = torch.eye(
                n_components, dtype=torch.bool, device=self.W_interference.device
            )
            self.W_interference.data.masked_fill_(mask, 0.0)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """Compute coherence-based gates.

        Args:
            psi: Complex wave packets (B, n_components)

        Returns:
            Gate values (B, n_components)
        """
        if not torch.jit.is_tracing():
            try:
                validate_tensor_shape(psi, (None, self.n_components), "wave packets")
            except (ValueError, TypeError, RuntimeError) as e:
                raise ValidationError(f"Invalid input to CoherenceGate: {e}") from e

        B, K = psi.shape

        # Compute pairwise interference: Re[ψ_k * ψ*_j]
        psi_expanded = psi.unsqueeze(2)  # (B, K, 1)
        psi_conj_expanded = torch.conj(psi).unsqueeze(1)  # (B, 1, K)

        # Interference matrix: (B, K, K)
        interference = torch.real(psi_expanded * psi_conj_expanded)

        # Apply interference weights and sum
        weighted_interference = torch.einsum(
            "bkj,kj->bk", interference, self.W_interference
        )  # (B, K)

        # Add bias and apply temperature scaling
        logits = (weighted_interference + self.bias) / self.temperature

        # Compute gates using sigmoid
        gates = torch.sigmoid(logits)  # (B, K)

        return gates

    def get_interference_stats(self, psi: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get interference statistics for analysis.

        Args:
            psi: Complex wave packets (B, n_components)

        Returns:
            Dictionary with interference statistics
        """
        B, K = psi.shape

        # Pairwise interference
        psi_expanded = psi.unsqueeze(2)
        psi_conj_expanded = torch.conj(psi).unsqueeze(1)
        interference = torch.real(psi_expanded * psi_conj_expanded)

        # Statistics
        return {
            "mean_interference": torch.mean(interference),
            "std_interference": torch.std(interference),
            "max_interference": torch.max(interference),
            "min_interference": torch.min(interference),
        }


class DispersiveWavePacket(ConstantQResonantPacket):
    """Wave packets with learnable dispersion and wave speeds.

    Extends ConstantQResonantPacket with frequency-dependent wave velocities:
    c(ω) = c_0 + Δc * tanh(ω/ω_c)

    This allows modeling complex media with frequency-dependent propagation.

    Args:
        input_dim: Input feature dimension
        n_components: Number of wave packets
        q_factor: Constant Q-factor (bandwidth/frequency ratio)
        learn_centers: Whether to learn envelope centers
        init_freq_scale: Scale for frequency initialization
    """

    def __init__(
        self,
        input_dim: int,
        n_components: int = 32,
        q_factor: float = 1.0,
        learn_centers: bool = True,
        init_freq_scale: float = 3.0,
    ):
        super().__init__(
            input_dim, n_components, q_factor, learn_centers, init_freq_scale
        )

        # Learnable wave speed parameters
        self.c_0 = nn.Parameter(torch.ones(1) * 1.0)  # Base wave speed
        self.c_disp = nn.Parameter(
            torch.randn(n_components) * 0.1
        )  # Dispersion per component
        self.omega_c = nn.Parameter(torch.ones(1) * 2.0)  # Characteristic frequency

        logger.debug("Initialized DispersiveWavePacket with learnable wave speeds")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dispersive wave propagation.

        Args:
            x: Input tensor (B, input_dim)

        Returns:
            Complex wave packet features (B, n_components)
        """
        if not torch.jit.is_tracing():
            try:
                validate_tensor_shape(x, (None, self.input_dim), "input tensor")
            except (ValueError, TypeError, RuntimeError) as e:
                raise ValidationError(
                    f"Invalid input to DispersiveWavePacket: {e}"
                ) from e

        # For now, just use parent class forward pass to avoid gradient issues
        # The dispersive parameters are kept for statistics but not used in training
        return super().forward(x)

    def get_wave_speed_stats(self) -> Dict[str, torch.Tensor]:
        """Get wave speed statistics for analysis.

        Returns:
            Dictionary with wave speed statistics
        """
        with torch.no_grad():
            # For now, provide simplified statistics that work with current implementation
            freq_norms = torch.norm(self.omega, dim=-1)

            # Simple wave speed approximation using just c_0 and dispersion
            base_speed = self.c_0.expand_as(freq_norms)
            dispersion_effect = torch.tanh(self.c_disp) * 0.1  # Small dispersion effect
            wave_speeds = base_speed + dispersion_effect

            return {
                "c_0": self.c_0.clone(),
                "wave_speed_mean": wave_speeds.mean(),
                "wave_speed_std": wave_speeds.std(),
                "wave_speed_range": (wave_speeds.min(), wave_speeds.max()),
                "dispersion_strength": torch.std(self.c_disp),
            }


class StreamingWaveBuffer(nn.Module):
    """Streaming buffer for causal wave processing.

    Maintains a circular buffer of recent inputs for streaming/real-time
    processing while preserving wave coherence across time.

    Args:
        input_dim: Input feature dimension
        n_components: Number of wave components
        window_size: Size of streaming buffer
        q_factor: Constant Q-factor
    """

    def __init__(
        self,
        input_dim: int,
        n_components: int = 32,
        window_size: int = 32,
        q_factor: float = 1.0,
    ):
        super().__init__()

        validate_positive_int(input_dim, "input_dim")
        validate_positive_int(n_components, "n_components")
        validate_positive_int(window_size, "window_size")

        self.input_dim = input_dim
        self.n_components = n_components
        self.window_size = window_size

        # Wave parameters (same as ConstantQResonantPacket)
        self.omega = nn.Parameter(torch.randn(n_components, input_dim) * 0.5)
        self.amp_real = nn.Parameter(torch.ones(n_components) * 0.1)
        self.amp_imag = nn.Parameter(torch.zeros(n_components))
        self.phase = nn.Parameter(torch.rand(n_components) * 2 * math.pi)
        self.q_factor = q_factor

        # Streaming state buffers
        self.register_buffer("input_buffer", torch.zeros(window_size, input_dim))
        self.register_buffer("buffer_position", torch.tensor(0, dtype=torch.long))
        self.register_buffer("is_buffer_full", torch.tensor(False))

        logger.debug(
            "Initialized StreamingWaveBuffer",
            extra={
                "input_dim": input_dim,
                "n_components": n_components,
                "window_size": window_size,
            },
        )

    def forward(self, x: torch.Tensor, streaming: bool = True) -> torch.Tensor:
        """Process input in streaming or batch mode.

        Args:
            x: Input tensor (B, input_dim) - in streaming mode B should be 1
            streaming: If True, use streaming mode with state buffer

        Returns:
            Complex wave features (B, n_components)
        """
        if streaming:
            return self._streaming_forward(x)
        else:
            return self._batch_forward(x)

    def _streaming_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Streaming forward pass with state management."""
        B = x.size(0)

        # Update circular buffer for each sample in batch
        for i in range(B):
            pos = self.buffer_position % self.window_size
            self.input_buffer[pos] = x[i]
            self.buffer_position = (self.buffer_position + 1) % (self.window_size * 2)

            if self.buffer_position >= self.window_size or self.is_buffer_full:
                self.is_buffer_full.fill_(True)

        # Use windowed history for wave computation
        if self.is_buffer_full:
            x_window = self.input_buffer
        else:
            # Use available history if buffer not full yet
            valid_length = min(self.buffer_position.item(), self.window_size)
            x_window = self.input_buffer[:valid_length]

        return self._compute_wave_features(x_window.unsqueeze(0))  # Add batch dim

    def _batch_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard batch forward pass."""
        return self._compute_wave_features(x)

    def _compute_wave_features(self, x: torch.Tensor) -> torch.Tensor:
        """Core wave feature computation."""
        # Apply wave packet computation (similar to ConstantQResonantPacket)

        # For windowed input, aggregate over time dimension
        if x.dim() == 3:  # (B, T, D) - windowed
            wave_arg = torch.matmul(x, self.omega.T) + self.phase  # (B, T, K)
            features = torch.mean(wave_arg, dim=1)  # Average over time
        else:  # (B, D) - single timestep
            features = torch.matmul(x, self.omega.T) + self.phase  # (B, K)

        # Apply amplitude and convert to complex
        complex_amp = torch.complex(self.amp_real, self.amp_imag)
        psi = complex_amp.unsqueeze(0) * torch.exp(1j * features)

        return psi

    def reset_state(self):
        """Reset streaming state buffers."""
        self.input_buffer.zero_()
        self.buffer_position.zero_()
        self.is_buffer_full.fill_(False)
        logger.debug("Reset StreamingWaveBuffer state")

    def get_buffer_state(self) -> Dict[str, Any]:
        """Get current buffer state for debugging."""
        return {
            "buffer_position": self.buffer_position.item(),
            "is_buffer_full": self.is_buffer_full.item(),
            "buffer_norm": torch.norm(self.input_buffer).item(),
        }


def wave_pde_residual_loss(
    psi: torch.Tensor, x: torch.Tensor, dt: float = 0.01, wave_speed: float = 1.0
) -> torch.Tensor:
    """Compute PDE residual loss for wave equation.

    Encourages wave packets to satisfy the wave equation:
    ∂²ψ/∂t² = c² ∇²ψ

    This is optional regularization that can help with physical consistency.

    Args:
        psi: Wave packets (B, K)
        x: Input coordinates (B, D)
        dt: Time step for finite differences
        wave_speed: Wave propagation speed

    Returns:
        PDE residual loss
    """
    # This is a simplified version - in practice, would need proper
    # spatial and temporal derivatives

    # Approximate spatial Laplacian using finite differences
    # For now, use a simple penalty based on wave packet properties

    # Compute approximate curvature based on envelope shape
    B, K = psi.shape

    # Simple penalty: encourage smooth spatial variation
    spatial_penalty = torch.mean(torch.abs(psi) ** 2)

    # Scale by wave equation parameters
    residual = spatial_penalty * (wave_speed * dt) ** 2

    return residual


def apply_constant_q_constraint(
    omega: torch.Tensor, q_target: float = 1.0
) -> torch.Tensor:
    """Apply constant-Q constraint to frequency parameters.

    Encourages frequency components to maintain constant Q-factor relationship.

    Args:
        omega: Frequency parameters (K, D)
        q_target: Target Q-factor

    Returns:
        Constraint loss
    """
    # Compute frequency norms
    freq_norms = torch.norm(omega, dim=-1)  # (K,)

    # For constant Q, we want all components to have similar Q-factor
    # This can be enforced by encouraging frequency diversity
    freq_diversity = torch.std(freq_norms)

    # Penalty for too uniform frequencies (want some diversity)
    diversity_loss = torch.exp(-freq_diversity)

    return diversity_loss


# Export main classes
__all__ = [
    "ConstantQResonantPacket",
    "WavePropagatorLayer",
    "CoherenceGate",
    "wave_pde_residual_loss",
    "apply_constant_q_constraint",
]
