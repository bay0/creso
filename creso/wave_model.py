"""
Enhanced CReSO model with wave physics.

Integrates wave-principled components with the existing CReSO architecture:
- Constant-Q resonant packets for spectral processing
- Unitary wave propagation layers
- Coherence-based interference gating
- Physics-aware regularization
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from .config import CReSOConfiguration
from .wave_physics import (
    ConstantQResonantPacket,
    WavePropagatorLayer,
    CoherenceGate,
    DispersiveWavePacket,
    StreamingWaveBuffer,
    wave_pde_residual_loss,
    apply_constant_q_constraint,
)
from .logging import get_logger
from .validation import validate_tensor_shape
from .exceptions import ValidationError

# Regularization constants
MIN_PAIRWISE_DISTANCE = 1e-6
MAX_DISPERSION_PENALTY = 1000.0

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None

logger = get_logger(__name__)


class CReSOWaveModel(nn.Module):
    """Enhanced CReSO model with wave physics.

    Combines wave-principled spectral processing with geometric processing:

    1. Wave Spectral Path:
       - Constant-Q resonant packets generate localized wave features
       - Unitary propagation layers apply wave evolution dynamics
       - Multiple propagation steps simulate wave equation behavior

    2. Coherence Gating:
       - Interference-based gating combines wave and geometric features
       - Adaptive selection based on phase relationships

    3. Geometric Path:
       - Traditional MLP processing for non-wave patterns
       - Provides complementary geometric reasoning

    Args:
        config: Model configuration
        use_wave_physics: Enable wave physics components
        n_propagation_steps: Number of wave propagation layers
    """

    def __init__(
        self,
        config: CReSOConfiguration,
        use_wave_physics: bool = True,
        n_propagation_steps: int = 2,
    ):
        super().__init__()
        self.config = config
        self.use_wave_physics = use_wave_physics
        self.n_propagation_steps = n_propagation_steps

        # Extract configuration
        self.input_dim = config.architecture.input_dim
        n_components = config.architecture.n_components
        geom_hidden = config.architecture.geometric_hidden_dim

        # Get wave physics config if available
        wave_config = getattr(config, "wave_physics", None)
        q_factor = wave_config.q_factor if wave_config else 1.0
        use_coherence_gate = wave_config.use_coherence_gate if wave_config else True

        logger.info(
            "Initializing CReSOWaveModel",
            extra={
                "input_dim": self.input_dim,
                "n_components": n_components,
                "use_wave_physics": use_wave_physics,
                "n_propagation_steps": n_propagation_steps,
                "q_factor": q_factor,
            },
        )

        if use_wave_physics:
            # Wave spectral processing path
            use_dispersive = getattr(wave_config, "use_dispersive_packets", False)

            if use_dispersive:
                self.wave_packets = DispersiveWavePacket(
                    input_dim=self.input_dim,
                    n_components=n_components,
                    q_factor=q_factor,
                    learn_centers=config.architecture.learn_envelope_centers,
                    init_freq_scale=config.architecture.initial_frequency_scale,
                )
                logger.info("Using dispersive wave packets")
            else:
                self.wave_packets = ConstantQResonantPacket(
                    input_dim=self.input_dim,
                    n_components=n_components,
                    q_factor=q_factor,
                    learn_centers=config.architecture.learn_envelope_centers,
                    init_freq_scale=config.architecture.initial_frequency_scale,
                )
                logger.info("Using standard constant-Q wave packets")

            # Multiple wave propagation layers
            self.propagation_layers = nn.ModuleList(
                [
                    WavePropagatorLayer(n_features=n_components)
                    for _ in range(n_propagation_steps)
                ]
            )

            # Coherence-based gating
            if use_coherence_gate:
                self.coherence_gate = CoherenceGate(n_components=n_components)
            else:
                self.coherence_gate = None

            logger.info(
                f"Initialized wave physics with {n_propagation_steps} propagation steps"
            )

        else:
            # Fall back to standard spectral layer
            from .layers import WaveResonanceLayer

            self.wave_layer = WaveResonanceLayer(
                input_dim=self.input_dim,
                n_components=n_components,
                localized=config.architecture.use_localized_envelopes,
                learn_centers=config.architecture.learn_envelope_centers,
                init_freq_scale=config.architecture.initial_frequency_scale,
            )

        # Geometric processing path (unchanged from base CReSO)
        self.geom_net = nn.Sequential(
            nn.Linear(self.input_dim, geom_hidden), nn.ReLU(), nn.Linear(geom_hidden, 1)
        )

        # Output combination layer
        if use_wave_physics and self.coherence_gate is not None:
            # Coherence gate combines wave and geometric features
            self.output_layer = nn.Linear(2, 1)  # z_spec + z_geom -> output
        else:
            # Standard gating mechanism
            self.gate = nn.Sequential(
                nn.Linear(self.input_dim + 2, 1),  # x + z_spec + z_geom -> alpha
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor, train_mode: bool = True) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, Any],
    ]:
        """Forward pass with wave physics.

        Args:
            x: Input tensor (B, input_dim)
            train_mode: Whether in training mode

        Returns:
            z: Final output (B, 1)
            z_spec: Spectral path output (B, 1)
            z_geom: Geometric path output (B, 1)
            alpha: Gating weights (B, 1) or coherence gates (B, K)
            wave_info: Dictionary with wave physics information
        """
        if not torch.jit.is_tracing():
            try:
                validate_tensor_shape(x, (None, self.input_dim), "input tensor")
            except Exception as e:
                raise ValidationError(f"Invalid input to CReSOWaveModel: {e}")

        wave_info = {}

        if self.use_wave_physics:
            # Wave spectral path
            psi = self.wave_packets(x)  # (B, K) complex
            wave_info["initial_energy"] = torch.mean(torch.abs(psi) ** 2).item()

            # Apply wave propagation steps
            psi_evolved = psi
            for i, prop_layer in enumerate(self.propagation_layers):
                psi_new = prop_layer(psi_evolved)

                # Track energy conservation
                energy_loss = prop_layer.energy_conservation_loss(psi_evolved, psi_new)
                wave_info[f"energy_loss_step_{i}"] = energy_loss.item()

                psi_evolved = psi_new

            wave_info["final_energy"] = torch.mean(torch.abs(psi_evolved) ** 2).item()

            # Extract spectral features (magnitude)
            spectral_features = torch.abs(psi_evolved)  # (B, K) real

            if self.coherence_gate is not None:
                # Coherence-based gating
                coherence_gates = self.coherence_gate(psi_evolved)  # (B, K)
                gated_spectral = coherence_gates * spectral_features  # (B, K)

                # Aggregate spectral features
                z_spec = torch.sum(gated_spectral, dim=-1, keepdim=True)  # (B, 1)

                # Store coherence information
                wave_info["coherence_stats"] = (
                    self.coherence_gate.get_interference_stats(psi_evolved)
                )
                wave_info["mean_gate_activation"] = torch.mean(coherence_gates).item()
                alpha = coherence_gates  # Return gates for analysis

            else:
                # Simple aggregation without coherence
                z_spec = torch.mean(spectral_features, dim=-1, keepdim=True)  # (B, 1)
                alpha = torch.ones_like(z_spec)  # Dummy gates

        else:
            # Standard spectral processing (fallback)
            freq_dropout_p = (
                self.config.training.frequency_dropout_probability
                if train_mode
                else 0.0
            )
            z_spec, C, S, mask = self.wave_layer(x, freq_dropout_p)
            wave_info = {"spectral_components": (C, S, mask)}
            alpha = torch.ones_like(z_spec)  # Dummy gates

        # Geometric path (unchanged)
        z_geom = self.geom_net(x)  # (B, 1)

        # Combine paths
        if self.use_wave_physics and self.coherence_gate is not None:
            # Coherence-gated combination
            combined = torch.cat([z_spec, z_geom], dim=-1)  # (B, 2)
            z = self.output_layer(combined)  # (B, 1)
        else:
            # Standard gating
            gate_input = torch.cat([x, z_spec, z_geom], dim=-1)  # (B, D+2)
            alpha = self.gate(gate_input)  # (B, 1)
            z = alpha * z_spec + (1 - alpha) * z_geom  # (B, 1)

        return z, z_spec, z_geom, alpha, wave_info

    def regularization(
        self,
        l2_freq: Optional[float] = None,
        group_l1: Optional[float] = None,
        center_disp: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute regularization loss (compatible with base CReSO interface).

        Args:
            l2_freq: L2 penalty on frequencies (uses config if None)
            group_l1: Group L1 penalty on amplitudes (uses config if None)
            center_disp: Center dispersion penalty (uses config if None)

        Returns:
            Regularization loss
        """
        # Get regularization weights from config
        l2_freq = l2_freq or self.config.regularization.l2_frequency_penalty
        group_l1 = group_l1 or self.config.regularization.group_l1_amplitude_penalty
        center_disp = (
            center_disp or self.config.regularization.center_dispersion_penalty
        )

        if not self.use_wave_physics:
            # Fall back to standard model regularization
            return self.wave_layer.regularization(l2_freq, group_l1, center_disp)

        total_reg = torch.tensor(0.0, device=next(self.parameters()).device)

        # L2 regularization on frequencies
        if l2_freq > 0:
            freq_l2 = torch.sum(self.wave_packets.omega**2)
            total_reg += l2_freq * freq_l2

        # Group L1 regularization on amplitudes
        if group_l1 > 0:
            amp_norms = torch.sqrt(
                self.wave_packets.amp_real**2 + self.wave_packets.amp_imag**2
            )
            amp_l1 = torch.sum(amp_norms)
            total_reg += group_l1 * amp_l1

        # Center dispersion regularization (encourage diverse centers)
        if center_disp > 0 and hasattr(self.wave_packets, "centers"):
            disp_penalty = self._compute_center_dispersion_penalty(self.wave_packets.centers)
            total_reg += center_disp * disp_penalty

        return total_reg

    def _compute_center_dispersion_penalty(self, centers: torch.Tensor) -> torch.Tensor:
        """Compute center dispersion penalty to encourage diverse centers.

        Args:
            centers: Center positions tensor of shape (K, D)

        Returns:
            Dispersion penalty value
        """
        # Calculate pairwise distances between centers
        pairwise_dist = torch.cdist(centers, centers)  # (K, K)

        # Penalize small distances using exponential penalty
        # Clamp distances to avoid numerical instability
        clamped_dist = torch.clamp(pairwise_dist, min=MIN_PAIRWISE_DISTANCE)

        # Compute penalty: sum of exponential decay minus diagonal elements
        disp_penalty = torch.sum(torch.exp(-clamped_dist)) - centers.shape[0]

        # Clamp penalty to prevent explosion
        return torch.clamp(disp_penalty, max=MAX_DISPERSION_PENALTY)

    def compute_wave_regularization_loss(
        self,
        x: torch.Tensor,
        wave_info: Dict[str, Any],
        energy_weight: float = 0.1,
        pde_weight: float = 0.01,
        q_constraint_weight: float = 0.01,
    ) -> torch.Tensor:
        """Compute wave physics regularization losses.

        Args:
            x: Input tensor
            wave_info: Wave information from forward pass
            energy_weight: Weight for energy conservation loss
            pde_weight: Weight for PDE residual loss
            q_constraint_weight: Weight for constant-Q constraint

        Returns:
            Total regularization loss
        """
        if not self.use_wave_physics:
            return torch.tensor(0.0, device=x.device)

        total_loss = torch.tensor(0.0, device=x.device)

        # Energy conservation loss
        if "energy_loss_step_0" in wave_info:
            energy_losses = [
                wave_info[k]
                for k in wave_info.keys()
                if k.startswith("energy_loss_step_")
            ]
            total_energy_loss = sum(energy_losses)
            total_loss += energy_weight * total_energy_loss

        # PDE residual loss (optional)
        if pde_weight > 0:
            psi = self.wave_packets(x)
            pde_loss = wave_pde_residual_loss(psi, x)
            total_loss += pde_weight * pde_loss

        # Constant-Q constraint
        if q_constraint_weight > 0:
            q_loss = apply_constant_q_constraint(self.wave_packets.omega)
            total_loss += q_constraint_weight * q_loss

        return total_loss

    def get_wave_component_info(self) -> Dict[str, Any]:
        """Get information about wave components for analysis.

        Returns:
            Dictionary with wave component statistics
        """
        if not self.use_wave_physics:
            return {"wave_physics_enabled": False}

        info = {"wave_physics_enabled": True}

        # Frequency component analysis
        omega = self.wave_packets.omega
        freq_norms = torch.norm(omega, dim=-1)
        info.update(
            {
                "n_components": self.wave_packets.n_components,
                "q_factor": self.wave_packets.q_factor,
                "freq_norm_mean": freq_norms.mean().item(),
                "freq_norm_std": freq_norms.std().item(),
                "freq_norm_range": (freq_norms.min().item(), freq_norms.max().item()),
            }
        )

        # Q-factor analysis
        q_factors = self.wave_packets.get_q_factors()
        info.update(
            {
                "q_actual_mean": q_factors.mean().item(),
                "q_actual_std": q_factors.std().item(),
            }
        )

        # Propagation parameters
        if self.propagation_layers:
            tau_values = [layer.tau.item() for layer in self.propagation_layers]
            info.update(
                {
                    "propagation_steps": len(self.propagation_layers),
                    "tau_values": tau_values,
                    "tau_mean": sum(tau_values) / len(tau_values),
                }
            )

        return info

    def prune_wave_components(
        self, threshold_percentile: float = 50.0
    ) -> Dict[str, int]:
        """Prune wave components based on amplitude magnitude.

        Args:
            threshold_percentile: Percentile threshold for pruning

        Returns:
            Dictionary with pruning statistics
        """
        if not self.use_wave_physics:
            logger.warning("Cannot prune components: wave physics not enabled")
            return {"n_pruned": 0, "total_components": 0}

        n_pruned = self.wave_packets.prune_components(threshold_percentile)
        total_components = self.wave_packets.n_components

        logger.info(f"Pruned {n_pruned}/{total_components} wave components")
        return {
            "n_pruned": n_pruned,
            "total_components": total_components,
            "sparsity": n_pruned / total_components,
        }

    def get_spectral_info(self) -> Dict[str, torch.Tensor]:
        """Get spectral information for compatibility with base CReSO.

        Returns:
            Dictionary with spectral component information
        """
        if not self.use_wave_physics:
            # Fall back to standard spectral layer
            return self.wave_layer.get_spectral_info()

        # Extract wave packet information
        omega_norms = torch.norm(self.wave_packets.omega, dim=-1)
        amplitudes = torch.sqrt(
            self.wave_packets.amp_real**2 + self.wave_packets.amp_imag**2
        )

        return {
            "freq_magnitudes": omega_norms,
            "amp_magnitudes": amplitudes,
        }

    def to_onnx(
        self,
        filepath: str,
        example_input: Optional[torch.Tensor] = None,
        opset: int = 17,
        optimize: bool = True,
        return_all_outputs: bool = False,
        verify_model: bool = True,
    ) -> None:
        """Export model to ONNX format with advanced options.

        Args:
            filepath: Path to save ONNX model
            example_input: Example input for export (generates random if None)
            opset: ONNX opset version (17 for latest features)
            optimize: Whether to optimize the ONNX model
            return_all_outputs: Whether to export all outputs or just main output
            verify_model: Whether to verify the exported model
        """
        import os
        from typing import Tuple

        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX export requires: pip install onnx onnxruntime")

        self.eval()

        if example_input is None:
            example_input = torch.randn(1, self.input_dim)

        # Create wrapper based on output requirements
        if return_all_outputs:

            class CReSOWaveModelONNXFullWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(
                    self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                    z, z_spec, z_geom, alpha, _ = self.model(x, train_mode=False)
                    return z, z_spec, z_geom, alpha

            wrapper = CReSOWaveModelONNXFullWrapper(self)
            output_names = [
                "output",
                "spectral_output",
                "geometric_output",
                "gate_weights",
            ]
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
                "spectral_output": {0: "batch_size"},
                "geometric_output": {0: "batch_size"},
                "gate_weights": {0: "batch_size"},
            }
        else:

            class CReSOWaveModelONNXWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    z, _, _, _, _ = self.model(x, train_mode=False)
                    return z

            wrapper = CReSOWaveModelONNXWrapper(self)
            output_names = ["output"]
            dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

        # Ensure directory exists
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )

        # Export with advanced options
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                example_input,
                filepath,
                export_params=True,
                opset_version=opset,
                do_constant_folding=optimize,
                input_names=["input"],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False,
                keep_initializers_as_inputs=False,
                strip_doc_string=True,
            )

        # Optimize ONNX model if requested
        if optimize:
            try:
                model_onnx = onnx.load(filepath)

                # Basic optimization
                from onnx import optimizer

                optimized_model = optimizer.optimize(model_onnx)
                onnx.save(optimized_model, filepath)

                logger.info("Applied ONNX optimization to CReSOWaveModel")
            except Exception as e:
                logger.warning(f"ONNX optimization failed: {e}")

        # Verify exported model if requested
        if verify_model:
            try:
                # Load and verify the ONNX model
                onnx_model = onnx.load(filepath)
                onnx.checker.check_model(onnx_model)

                # Test inference with ONNX Runtime
                try:
                    ort_session = ort.InferenceSession(
                        filepath, providers=["CPUExecutionProvider"]
                    )

                    # Run inference test
                    ort_inputs = {"input": example_input.numpy()}
                    ort_outputs = ort_session.run(None, ort_inputs)

                    # Compare with PyTorch output
                    with torch.no_grad():
                        torch_output = wrapper(example_input)
                        if isinstance(torch_output, tuple):
                            torch_output = torch_output[0]  # Main output

                        diff = abs(torch_output.numpy() - ort_outputs[0]).max()
                        if diff < 1e-5:
                            logger.info(
                                f"ONNX CReSOWaveModel verification passed (max diff: {diff:.2e})"
                            )
                        else:
                            logger.warning(
                                f"ONNX CReSOWaveModel verification: large difference {diff:.2e}"
                            )
                finally:
                    # Ensure session is properly cleaned up
                    if 'ort_session' in locals():
                        del ort_session

            except Exception as e:
                logger.warning(f"ONNX CReSOWaveModel verification failed: {e}")

        logger.info(f"Exported optimized ONNX CReSOWaveModel to: {filepath}")

    def to_torchscript(
        self,
        filepath: str,
        example_input: Optional[torch.Tensor] = None,
        optimize: bool = True,
        quantize: bool = False,
        return_all_outputs: bool = False,
    ) -> None:
        """Export model to TorchScript format with optimization options.

        Args:
            filepath: Path to save TorchScript model
            example_input: Example input for tracing (generates random if None)
            optimize: Whether to optimize the traced model
            quantize: Whether to apply dynamic quantization
            return_all_outputs: Whether to return all outputs or just main output
        """
        import os

        # Switch to evaluation mode
        self.eval()

        if example_input is None:
            example_input = torch.randn(1, self.input_dim)

        # Create wrapper based on output requirements
        if return_all_outputs:

            class CReSOWaveModelFullWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(
                    self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                    z, z_spec, z_geom, alpha, _ = self.model(x, train_mode=False)
                    return z, z_spec, z_geom, alpha

            wrapper = CReSOWaveModelFullWrapper(self)
        else:

            class CReSOWaveModelWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    z, _, _, _, _ = self.model(x, train_mode=False)
                    return z

            wrapper = CReSOWaveModelWrapper(self)

        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapper, example_input, strict=False)

        # Optimize if requested
        if optimize:
            traced_model = torch.jit.optimize_for_inference(traced_model)
            logger.info("Applied TorchScript optimization for CReSOWaveModel")

        # Apply dynamic quantization if requested
        if quantize:
            try:
                traced_model = torch.quantization.quantize_dynamic(
                    traced_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Applied dynamic quantization to CReSOWaveModel")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")

        # Ensure directory exists
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )

        # Save with metadata
        extra_files = {
            "config.json": str(self.config.__dict__),
            "input_shape.txt": str(list(example_input.shape)),
            "wave_physics.txt": f"CReSOWaveModel exported with wave_physics={self.use_wave_physics}",
            "model_info.txt": f"CReSOWaveModel exported with optimize={optimize}, quantize={quantize}",
        }

        traced_model.save(filepath, _extra_files=extra_files)
        logger.info(f"Exported optimized TorchScript CReSOWaveModel to: {filepath}")


class CReSOStreamingModel(nn.Module):
    """Streaming variant of CReSO with causal/online processing.

    Designed for real-time applications where inputs arrive sequentially
    and processing must be causal (no future information).

    Args:
        config: Model configuration
        window_size: Size of streaming buffer
        use_dispersive: Whether to use dispersive wave packets
    """

    def __init__(
        self,
        config: CReSOConfiguration,
        window_size: int = 32,
        use_dispersive: bool = False,
    ):
        super().__init__()
        self.config = config
        self.window_size = window_size
        self.use_dispersive = use_dispersive

        # Extract configuration
        self.input_dim = config.architecture.input_dim
        n_components = config.architecture.n_components
        geom_hidden = config.architecture.geometric_hidden_dim

        # Get wave physics config
        wave_config = getattr(config, "wave_physics", None)
        q_factor = wave_config.q_factor if wave_config else 1.0

        logger.info(
            "Initializing CReSOStreamingModel",
            extra={
                "input_dim": self.input_dim,
                "n_components": n_components,
                "window_size": window_size,
                "use_dispersive": use_dispersive,
            },
        )

        # Streaming wave processing
        if use_dispersive:
            # Use dispersive wave packets (but in streaming buffer)
            self.wave_buffer = StreamingWaveBuffer(
                input_dim=self.input_dim,
                n_components=n_components,
                window_size=window_size,
                q_factor=q_factor,
            )
        else:
            # Standard streaming buffer
            self.wave_buffer = StreamingWaveBuffer(
                input_dim=self.input_dim,
                n_components=n_components,
                window_size=window_size,
                q_factor=q_factor,
            )

        # Geometric processing (unchanged)
        self.geom_net = nn.Sequential(
            nn.Linear(self.input_dim, geom_hidden), nn.ReLU(), nn.Linear(geom_hidden, 1)
        )

        # Output combination
        self.output_layer = nn.Linear(2, 1)  # z_spec + z_geom -> output

        # Streaming state
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def forward(
        self, x: torch.Tensor, streaming: bool = True, reset_state: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Forward pass with streaming support.

        Args:
            x: Input tensor (B, input_dim) - for streaming B should be 1
            streaming: Use streaming mode with state buffer
            reset_state: Reset streaming state before processing

        Returns:
            z: Final output (B, 1)
            z_spec: Spectral path output (B, 1)
            z_geom: Geometric path output (B, 1)
            stream_info: Streaming information
        """
        if reset_state:
            self.reset_streaming_state()

        # Streaming wave processing
        psi = self.wave_buffer(x, streaming=streaming)  # (B, K) complex

        # Extract spectral features (magnitude)
        spectral_features = torch.abs(psi)  # (B, K) real
        z_spec = torch.mean(spectral_features, dim=-1, keepdim=True)  # (B, 1)

        # Geometric processing
        z_geom = self.geom_net(x)  # (B, 1)

        # Combine paths
        combined = torch.cat([z_spec, z_geom], dim=-1)  # (B, 2)
        z = self.output_layer(combined)  # (B, 1)

        # Update step count
        if streaming:
            self.step_count += x.size(0)

        # Streaming info
        stream_info = {
            "step_count": self.step_count.item(),
            "buffer_state": self.wave_buffer.get_buffer_state(),
            "spectral_magnitude_mean": torch.mean(spectral_features).item(),
        }

        return z, z_spec, z_geom, stream_info

    def reset_streaming_state(self):
        """Reset all streaming state."""
        self.wave_buffer.reset_state()
        self.step_count.zero_()
        logger.debug("Reset CReSOStreamingModel state")

    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming processing statistics."""
        return {
            "total_steps_processed": self.step_count.item(),
            "window_size": self.window_size,
            "buffer_state": self.wave_buffer.get_buffer_state(),
        }


# Backward compatibility: alias for standard usage
CReSOWaveModelLegacy = CReSOWaveModel


__all__ = [
    "CReSOWaveModel",
    "CReSOStreamingModel",
    "CReSOWaveModelLegacy",
]
