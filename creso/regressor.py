"""
Regression utilities for CReSO models.

Provides CReSORegressor for continuous target prediction.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Union, Optional, Dict
from pathlib import Path

from .config import CReSOConfiguration
from .trainer import CReSOTrainer
from .model import CReSOModel
from .wave_model import CReSOWaveModel
from .utils import as_tensor
from .logging import get_logger
from .validation import validate_tensor_2d
from .exceptions import ValidationError

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None

logger = get_logger(__name__)


class CReSORegressor:
    """CReSO model for regression tasks.

    A scikit-learn compatible regressor using CReSO architectures
    for continuous target prediction.

    Args:
        config: CReSO configuration object

    Example:
        >>> from creso import CReSORegressor, CReSOConfiguration
        >>> config = CReSOConfiguration(...)
        >>> regressor = CReSORegressor(config)
        >>> regressor.fit(X_train, y_train)
        >>> predictions = regressor.predict(X_test)
    """

    def __init__(self, config: CReSOConfiguration):
        self.config = config
        self.model = None
        self.trainer = None
        self.standardizer = None
        self._is_fitted = False

        logger.info("Initialized CReSORegressor")

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        sample_weight: Optional[Union[np.ndarray, torch.Tensor]] = None,
        standardize: bool = True,
    ) -> CReSORegressor:
        """Fit the CReSO regressor.

        Args:
            X: Training features (N, input_dim)
            y: Training targets (N,) - continuous values
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            sample_weight: Sample weights (not implemented yet)
            standardize: Whether to standardize features

        Returns:
            Self (for method chaining)
        """
        logger.info("Starting regressor training")

        # Convert inputs to tensors if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        elif not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        # Validate inputs
        X = validate_tensor_2d(X, "training features")
        y = self._validate_regression_targets(y, len(X))

        # Validate that both X_val and y_val are provided or both are None
        if (X_val is None) != (y_val is None):
            raise ValidationError(
                "Both X_val and y_val must be provided or both must be None"
            )

        if X_val is not None and y_val is not None:
            if isinstance(X_val, np.ndarray):
                X_val = torch.from_numpy(X_val).float()
            elif not isinstance(X_val, torch.Tensor):
                X_val = torch.tensor(X_val, dtype=torch.float32)
            X_val = validate_tensor_2d(X_val, "validation features")
            y_val = self._validate_regression_targets(y_val, len(X_val))

        # Create trainer with regression configuration
        self.trainer = CReSOTrainer(self._get_regression_config())

        # Train model - we'll create a regression-compatible trainer
        self.model, self.standardizer = self._fit_regression_model(
            X, y, X_val, y_val, standardize
        )

        self._is_fitted = True
        logger.info("Regressor training completed")
        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict continuous targets.

        Args:
            X: Input features (N, input_dim)

        Returns:
            Predicted targets (N,)
        """
        self._check_fitted()

        # Convert inputs to tensors if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        elif not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        X = validate_tensor_2d(X, "input features")

        device = next(self.model.parameters()).device
        X = as_tensor(X, device=device)

        if self.standardizer is not None:
            X = self.standardizer.transform(X)

        self.model.eval()
        with torch.no_grad():
            z, _, _, _, _ = self.model(X, train_mode=False)
            predictions = z.squeeze().cpu().numpy()

        return predictions

    def score(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        sample_weight: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> float:
        """Return R² coefficient of determination.

        Args:
            X: Test features
            y: True targets
            sample_weight: Sample weights (not implemented)

        Returns:
            R² score
        """
        predictions = self.predict(X)

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        # Calculate R²
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return float(r2)

    def get_regression_metrics(
        self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute comprehensive regression metrics.

        Args:
            X: Test features
            y: True targets

        Returns:
            Dictionary of regression metrics
        """
        predictions = self.predict(X)

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        # Calculate various metrics
        mse = np.mean((y - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - predictions))

        # R² (handle division by zero when all targets are the same)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Mean absolute percentage error (handle division by zero)
        mask = y != 0
        mape = (
            np.mean(np.abs((y[mask] - predictions[mask]) / y[mask])) * 100
            if np.any(mask)
            else np.inf
        )

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
            "n_samples": len(y),
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save the fitted regressor.

        Args:
            filepath: Path to save the regressor
        """
        self._check_fitted()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "model_class": type(self.model).__name__,
            "standardizer": self.standardizer,
            "is_fitted": self._is_fitted,
        }

        torch.save(save_dict, filepath)

        logger.info(f"Saved regressor to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> CReSORegressor:
        """Load a fitted regressor.

        Args:
            filepath: Path to saved regressor

        Returns:
            Loaded regressor
            
        Warning:
            Only load regressor files from trusted sources. This method uses
            torch.load with weights_only=False to support configuration objects.
        """
        save_dict = torch.load(filepath, weights_only=False, map_location="cpu")

        # Create regressor
        regressor = cls(save_dict["config"])

        # Recreate model
        device = torch.device(save_dict["config"].system.device)

        if save_dict["model_class"] == "CReSOWaveModel":
            model = CReSOWaveModel(
                save_dict["config"],
                use_wave_physics=True,
                n_propagation_steps=save_dict[
                    "config"
                ].wave_physics.n_propagation_steps,
            ).to(device)
        else:
            model = CReSOModel(save_dict["config"]).to(device)

        # Load model state
        model.load_state_dict(save_dict["model_state_dict"])

        regressor.model = model
        regressor.standardizer = save_dict["standardizer"]
        regressor._is_fitted = save_dict["is_fitted"]

        logger.info(f"Loaded regressor from {filepath}")
        return regressor

    def _validate_regression_targets(
        self, y: Union[np.ndarray, torch.Tensor], n_samples: int
    ) -> torch.Tensor:
        """Validate regression targets."""
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        elif isinstance(y, torch.Tensor):
            y = y.float()
        else:
            raise ValidationError("Targets must be numpy array or torch tensor")

        # Ensure correct shape
        if y.dim() == 1:
            pass  # (N,) is fine
        elif y.dim() == 2 and y.size(1) == 1:
            y = y.squeeze(1)  # (N, 1) -> (N,)
        else:
            raise ValidationError(
                f"Targets must be 1D or 2D with shape (N,) or (N, 1), got {y.shape}"
            )

        if len(y) != n_samples:
            raise ValidationError(
                f"Number of targets {len(y)} doesn't match number of samples {n_samples}"
            )

        # Check for NaN or infinite values
        if torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
            raise ValidationError("Targets contain NaN or infinite values")

        return y

    def _get_regression_config(self) -> CReSOConfiguration:
        """Get configuration modified for regression."""
        # For now, we'll use the same config but will modify the trainer to use regression losses
        return self.config

    def _fit_regression_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_val: Optional[torch.Tensor],
        y_val: Optional[torch.Tensor],
        standardize: bool,
    ):
        """Fit regression model with MSE loss."""
        # This is a simplified version - we would need to modify the trainer
        # to support different loss functions for regression

        # For now, create model directly and train with MSE loss
        device = torch.device(self.config.system.device)

        if self.config.wave_physics.enable_wave_physics:
            model = CReSOWaveModel(
                self.config,
                use_wave_physics=True,
                n_propagation_steps=self.config.wave_physics.n_propagation_steps,
            ).to(device)
        else:
            model = CReSOModel(self.config).to(device)

        # Simple training loop with MSE loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        criterion = nn.MSELoss()

        # Data standardization
        standardizer = None
        if standardize:
            from .utils import Standardizer

            standardizer = Standardizer()
            X = standardizer.fit_transform(X)
            if X_val is not None:
                X_val = standardizer.transform(X_val)

        X = X.to(device)
        y = y.to(device)
        if X_val is not None:
            X_val = X_val.to(device)
            y_val = y_val.to(device)

        model.train()
        epochs = self.config.training.max_epochs
        batch_size = self.config.training.batch_size

        logger.info(f"Training regression model for {epochs} epochs")

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Simple batch training
            n_samples = X.size(0)
            indices = torch.randperm(n_samples, device=device)

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                optimizer.zero_grad()

                z, _, _, _, _ = model(X_batch, train_mode=True)
                loss = criterion(z.squeeze(), y_batch)

                # Add model regularization
                loss = loss + model.regularization()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            val_loss = None
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    z_val, _, _, _, _ = model(X_val, train_mode=False)
                    val_loss = criterion(z_val.squeeze(), y_val).item()
                model.train()

            if epoch % 10 == 0 or epoch == epochs - 1:
                if val_loss is not None:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss/n_batches:.4f}, Val Loss={val_loss:.4f}"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss/n_batches:.4f}"
                    )

        model.eval()
        return model, standardizer

    def _check_fitted(self):
        """Check if the regressor is fitted."""
        if not self._is_fitted:
            raise ValidationError("Regressor must be fitted before making predictions")

    @property
    def feature_importance_(self):
        """Get feature importance based on spectral component magnitudes."""
        self._check_fitted()

        # Return amplitude magnitudes of spectral components as feature importance
        if hasattr(self.model, "wave_layer"):
            amp_magnitudes = self.model.wave_layer.get_amplitude_magnitudes()
            return amp_magnitudes.detach().cpu().numpy()
        else:
            logger.warning("Model does not have wave_layer for feature importance")
            return None

    def to_torchscript(
        self,
        filepath: str,
        example_input: Optional[torch.Tensor] = None,
        optimize: bool = True,
        quantize: bool = False,
    ) -> None:
        """Export regressor to TorchScript format.

        Args:
            filepath: Path to save TorchScript model
            example_input: Example input for tracing (generates random if None)
            optimize: Whether to optimize the traced model
            quantize: Whether to apply dynamic quantization
        """
        self._check_fitted()

        if example_input is None:
            example_input = torch.randn(1, self.config.architecture.input_dim)

        # Create wrapper for regression prediction
        class CReSORegressorWrapper(nn.Module):
            def __init__(self, regressor):
                super().__init__()
                self.regressor = regressor

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.regressor.model.eval()
                device = next(self.regressor.model.parameters()).device
                x = x.to(device)

                if self.regressor.standardizer is not None:
                    x = self.regressor.standardizer.transform(x)

                with torch.no_grad():
                    z, _, _, _, _ = self.regressor.model(x, train_mode=False)
                    return z.squeeze()

        wrapper = CReSORegressorWrapper(self)

        # Use the model's export method approach
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapper, example_input, strict=False)

        # Optimize if requested
        if optimize:
            traced_model = torch.jit.optimize_for_inference(traced_model)
            logger.info("Applied TorchScript optimization for CReSORegressor")

        # Apply dynamic quantization if requested
        if quantize:
            try:
                traced_model = torch.quantization.quantize_dynamic(
                    traced_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Applied dynamic quantization to CReSORegressor")
            except (RuntimeError, AttributeError, TypeError) as e:
                logger.warning(f"Quantization failed: {e}")

        # Ensure directory exists
        import os

        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )

        # Save with metadata
        extra_files = {
            "config.json": str(self.config.__dict__),
            "input_shape.txt": str(list(example_input.shape)),
            "model_info.txt": f"CReSORegressor exported with optimize={optimize}, quantize={quantize}",
        }

        traced_model.save(filepath, _extra_files=extra_files)
        logger.info(f"Exported CReSORegressor to TorchScript: {filepath}")

    def to_onnx(
        self,
        filepath: str,
        example_input: Optional[torch.Tensor] = None,
        opset: int = 17,
        optimize: bool = True,
        verify_model: bool = True,
    ) -> None:
        """Export regressor to ONNX format.

        Args:
            filepath: Path to save ONNX model
            example_input: Example input for export (generates random if None)
            opset: ONNX opset version
            optimize: Whether to optimize the ONNX model
            verify_model: Whether to verify the exported model
        """
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX export requires: pip install onnx onnxruntime")

        self._check_fitted()

        if example_input is None:
            example_input = torch.randn(1, self.config.architecture.input_dim)

        # Create wrapper for regression prediction
        class CReSORegressorONNXWrapper(nn.Module):
            def __init__(self, regressor):
                super().__init__()
                self.regressor = regressor

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.regressor.model.eval()
                device = next(self.regressor.model.parameters()).device
                x = x.to(device)

                if self.regressor.standardizer is not None:
                    x = self.regressor.standardizer.transform(x)

                with torch.no_grad():
                    z, _, _, _, _ = self.regressor.model(x, train_mode=False)
                    return z.squeeze()

        wrapper = CReSORegressorONNXWrapper(self)

        # Ensure directory exists
        import os

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
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                verbose=False,
                keep_initializers_as_inputs=False,
                strip_doc_string=True,
            )

        # Optimize ONNX model if requested
        if optimize and os.path.exists(filepath):
            try:
                model_onnx = onnx.load(filepath)

                # Basic optimization
                from onnx import optimizer

                optimized_model = optimizer.optimize(model_onnx)
                onnx.save(optimized_model, filepath)

                logger.info("Applied ONNX optimization to CReSORegressor")
            except (ImportError, RuntimeError, AttributeError) as e:
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

                        diff = abs(torch_output.numpy() - ort_outputs[0]).max()
                        if diff < 1e-5:
                            logger.info(
                                f"ONNX CReSORegressor verification passed (max diff: {diff:.2e})"
                            )
                        else:
                            logger.warning(
                                f"ONNX CReSORegressor verification: large difference {diff:.2e}"
                            )
                finally:
                    # Ensure session is properly cleaned up
                    if 'ort_session' in locals():
                        del ort_session

            except (ImportError, RuntimeError, ValueError, OSError) as e:
                logger.warning(f"ONNX CReSORegressor verification failed: {e}")

        logger.info(f"Exported CReSORegressor to ONNX: {filepath}")


# Utility functions for regression
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
