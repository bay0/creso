"""
Main CReSO model combining spectral and geometric processing paths.
"""

import os
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from .config import CReSOConfiguration
from .layers import WaveResonanceLayer
from .utils import Standardizer
from .logging import get_logger
from .validation import validate_tensor_shape
from .exceptions import ValidationError

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None

logger = get_logger(__name__)


class CReSOModel(nn.Module):
    """CReSO model with spectral and geometric paths.

    Combines a spectral processing path (WaveResonanceLayer) with a geometric
    path (MLP) using a learned gating mechanism.

    Args:
        config: Model configuration
    """

    def __init__(self, config: CReSOConfiguration):
        super().__init__()
        self.config = config

        # Extract configuration parameters
        self.input_dim = config.architecture.input_dim
        n_components = config.architecture.n_components
        localized = config.architecture.use_localized_envelopes
        learn_centers = config.architecture.learn_envelope_centers
        init_freq_scale = config.architecture.initial_frequency_scale
        geom_hidden = config.architecture.geometric_hidden_dim

        logger.info(
            "Initializing CReSOModel",
            extra={
                "input_dim": self.input_dim,
                "n_components": n_components,
                "localized": localized,
                "geom_hidden": geom_hidden,
            },
        )

        # Spectral processing path
        self.wave_layer = WaveResonanceLayer(
            input_dim=self.input_dim,
            n_components=n_components,
            localized=localized,
            learn_centers=learn_centers,
            init_freq_scale=init_freq_scale,
        )

        # Geometric processing path (simple MLP)
        self.geom_net = nn.Sequential(
            nn.Linear(self.input_dim, geom_hidden), nn.ReLU(), nn.Linear(geom_hidden, 1)
        )

        # Gating mechanism to combine paths
        # Takes both spectral and geometric features as input
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim + 2, 1),  # x + z_spec + z_geom -> alpha
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, train_mode: bool = True) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    ]:
        """Forward pass.

        Args:
            x: Input tensor (B, input_dim)
            train_mode: Whether in training mode (affects dropout)

        Returns:
            z: Final output (B, 1)
            z_spec: Spectral path output (B, 1)
            z_geom: Geometric path output (B, 1)
            alpha: Gating weights (B, 1)
            spectral_components: (C, S, mask) from wave layer
        """
        # Skip validation during TorchScript tracing
        if not torch.jit.is_tracing():
            try:
                validate_tensor_shape(x, (None, self.input_dim), "input tensor")
            except (ValueError, TypeError, RuntimeError) as e:
                raise ValidationError(f"Invalid input to CReSOModel: {e}") from e

        # Spectral path
        freq_dropout_p = (
            self.config.training.frequency_dropout_probability if train_mode else 0.0
        )

        z_spec, C, S, mask = self.wave_layer(x, freq_dropout_p=freq_dropout_p)

        # Geometric path
        z_geom = self.geom_net(x)

        # Gating mechanism
        gate_input = torch.cat([x, z_spec, z_geom], dim=1)  # (B, input_dim + 2)
        alpha = self.gate(gate_input)  # (B, 1)

        # Combine paths
        z = alpha * z_spec + (1 - alpha) * z_geom  # (B, 1)

        return z, z_spec, z_geom, alpha, (C, S, mask)

    def regularization(
        self,
        l2_freq: Optional[float] = None,
        group_l1: Optional[float] = None,
        center_disp: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute regularization loss.

        Args:
            l2_freq: L2 penalty on frequencies (uses config if None)
            group_l1: Group L1 penalty on amplitudes (uses config if None)
            center_disp: Center dispersion penalty (uses config if None)

        Returns:
            Regularization loss
        """
        if l2_freq is None:
            l2_freq = self.config.regularization.l2_frequency_penalty
        if group_l1 is None:
            group_l1 = self.config.regularization.group_l1_amplitude_penalty
        if center_disp is None:
            center_disp = self.config.regularization.center_dispersion_penalty

        return self.wave_layer.spectral_regularizers(l2_freq, group_l1, center_disp)

    def prune_spectral_components(
        self, top_k: Optional[int] = None, threshold: Optional[float] = None
    ) -> None:
        """Prune low-amplitude spectral components.

        Args:
            top_k: Keep only top K components
            threshold: Keep only components above threshold
        """
        self.wave_layer.prune_by_amplitude(top_k=top_k, threshold=threshold)

    def get_spectral_info(self) -> Dict[str, torch.Tensor]:
        """Get information about spectral components.

        Returns:
            Dictionary with frequency and amplitude information
        """
        return {
            "freq_magnitudes": self.wave_layer.get_frequency_magnitudes().detach(),
            "amp_magnitudes": self.wave_layer.get_amplitude_magnitudes().detach(),
            "frequencies": self.wave_layer.omega.detach().clone(),
            "phases": self.wave_layer.theta.detach().clone(),
            "cos_amps": self.wave_layer.a_c.detach().clone(),
            "sin_amps": self.wave_layer.a_s.detach().clone(),
        }

    def save(
        self,
        path: str,
        standardizer: Optional[Standardizer] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model with optional standardizer and extra data.

        Args:
            path: Save path
            standardizer: Data standardizer to save
            extra: Extra data to save
        """
        save_dict = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "standardizer": standardizer.state_dict() if standardizer else None,
            "extra": extra or {},
        }

        # Create directory if path contains a directory component
        dir_path = os.path.dirname(path)
        if dir_path:
            from .utils import safe_makedirs

            safe_makedirs(dir_path, exist_ok=True)

        torch.save(save_dict, path)

    @staticmethod
    def load(
        path: str, map_location: str = "cpu", weights_only: bool = True
    ) -> Tuple["CReSOModel", Optional[Standardizer], Dict[str, Any]]:
        """Load model with standardizer and extra data.

        Warning: Only load models from trusted sources. Loading untrusted
        models can execute arbitrary code.

        Args:
            path: Load path
            map_location: Device to map tensors to
            weights_only: Whether to use safe loading (True for external, False for internal)

        Returns:
            Loaded model, standardizer (if saved), and extra data
        """
        import warnings

        if not weights_only:
            # Security warning for unsafe loading
            warnings.warn(
                "Loading PyTorch models with weights_only=False from untrusted sources "
                "can execute arbitrary code. Only load models from trusted sources.",
                UserWarning,
                stacklevel=2,
            )
        if weights_only:
            from .config import (
                CReSOConfiguration,
                ModelArchitectureConfig,
                TrainingConfig,
                RegularizationConfig,
                FrequencySeedingConfig,
                SystemConfig,
                WavePhysicsConfig,
            )

            safe_classes = [
                CReSOConfiguration,
                ModelArchitectureConfig,
                TrainingConfig,
                RegularizationConfig,
                FrequencySeedingConfig,
                SystemConfig,
                WavePhysicsConfig,
            ]
            with torch.serialization.safe_globals(safe_classes):
                checkpoint = torch.load(
                    path, map_location=map_location, weights_only=True
                )
        else:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        config = checkpoint["config"]
        model = CReSOModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        standardizer = None
        if checkpoint.get("standardizer") is not None:
            standardizer = Standardizer()
            standardizer.load_state_dict(checkpoint["standardizer"])

        extra = checkpoint.get("extra", {})

        return model, standardizer, extra

    def to_torchscript(
        self,
        path: str,
        example_input: Optional[torch.Tensor] = None,
        optimize: bool = True,
        quantize: bool = False,
        return_all_outputs: bool = False,
    ) -> None:
        """Export model to TorchScript with optimization options.

        Args:
            path: Export path
            example_input: Example input for tracing (generates random if None)
            optimize: Whether to optimize the traced model
            quantize: Whether to apply dynamic quantization
            return_all_outputs: Whether to return all outputs or just main output
        """
        self.eval()

        if example_input is None:
            example_input = torch.randn(1, self.input_dim)

        # Create wrapper based on output requirements
        if return_all_outputs:

            class CReSOModelFullWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(
                    self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                    z, z_spec, z_geom, alpha, _ = self.model(x, train_mode=False)
                    return z, z_spec, z_geom, alpha

            wrapper = CReSOModelFullWrapper(self)
        else:

            class CReSOModelWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    z, _, _, _, _ = self.model(x, train_mode=False)
                    return z

            wrapper = CReSOModelWrapper(self)

        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapper, example_input, strict=False)

        # Optimize if requested
        if optimize:
            traced_model = torch.jit.optimize_for_inference(traced_model)
            logger.info("Applied TorchScript optimization")

        # Apply dynamic quantization if requested
        if quantize:
            try:
                traced_model = torch.quantization.quantize_dynamic(
                    traced_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Applied dynamic quantization")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")

        # Ensure directory exists
        dir_path = os.path.dirname(path) if os.path.dirname(path) else "."
        from .utils import safe_makedirs

        safe_makedirs(dir_path, exist_ok=True)

        # Save with metadata
        extra_files = {
            "config.json": str(self.config.__dict__),
            "input_shape.txt": str(list(example_input.shape)),
            "model_info.txt": f"CReSOModel exported with optimize={optimize}, quantize={quantize}",
        }

        traced_model.save(path, _extra_files=extra_files)
        logger.info(f"Exported optimized TorchScript model to: {path}")

    def to_onnx(
        self,
        path: str,
        example_input: Optional[torch.Tensor] = None,
        opset: int = 17,
        optimize: bool = True,
        return_all_outputs: bool = False,
        verify_model: bool = True,
    ) -> None:
        """Export model to ONNX format with advanced options.

        Args:
            path: Export path
            example_input: Example input for export (generates random if None)
            opset: ONNX opset version (17 for latest features)
            optimize: Whether to optimize the ONNX model
            return_all_outputs: Whether to export all outputs or just main output
            verify_model: Whether to verify the exported model
        """
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

            class CReSOModelONNXFullWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(
                    self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                    z, z_spec, z_geom, alpha, _ = self.model(x, train_mode=False)
                    return z, z_spec, z_geom, alpha

            wrapper = CReSOModelONNXFullWrapper(self)
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

            class CReSOModelONNXWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    z, _, _, _, _ = self.model(x, train_mode=False)
                    return z

            wrapper = CReSOModelONNXWrapper(self)
            output_names = ["output"]
            dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

        # Ensure directory exists
        dir_path = os.path.dirname(path) if os.path.dirname(path) else "."
        from .utils import safe_makedirs

        safe_makedirs(dir_path, exist_ok=True)

        # Export with advanced options
        try:
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    example_input,
                    path,
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
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise

        # Optimize ONNX model if requested
        if optimize and os.path.exists(path):
            try:
                model_onnx = onnx.load(path)

                # Basic optimization
                from onnx import optimizer

                optimized_model = optimizer.optimize(model_onnx)
                onnx.save(optimized_model, path)

                logger.info("Applied ONNX optimization")
            except (ImportError, RuntimeError, AttributeError) as e:
                logger.warning(f"ONNX optimization failed: {e}")

        # Verify exported model if requested
        if verify_model:
            try:
                # Load and verify the ONNX model
                onnx_model = onnx.load(path)
                onnx.checker.check_model(onnx_model)

                # Test inference with ONNX Runtime
                try:
                    ort_session = ort.InferenceSession(
                        path, providers=["CPUExecutionProvider"]
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
                                f"ONNX model verification passed (max diff: {diff:.2e})"
                            )
                        else:
                            logger.warning(
                                f"ONNX model verification: large difference {diff:.2e}"
                            )
                finally:
                    # Ensure session is properly cleaned up
                    if "ort_session" in locals():
                        del ort_session

            except (ImportError, RuntimeError, ValueError, OSError) as e:
                logger.warning(f"ONNX model verification failed: {e}")

        logger.info(f"Exported optimized ONNX model to: {path}")
