"""
Custom exception classes for CReSO package.

Provides structured error handling with meaningful error messages
and context information for debugging and monitoring.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class CReSOError(Exception):
    """Base exception class for all CReSO errors.

    Args:
        message: Human-readable error message
        error_code: Machine-readable error code
        context: Additional context information
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}

    def __str__(self) -> str:
        """String representation with error code and context."""
        parts = [f"{self.error_code}: {self.message}"]
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f" (Context: {context_str})")
        return "".join(parts)

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"{self.__class__.__name__}(message='{self.message}', error_code='{self.error_code}', context={self.context})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_class": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
        }

    def get_user_friendly_message(self) -> str:
        """Get a user-friendly error message with helpful suggestions."""
        base_message = self.message

        # Add context-specific suggestions
        suggestions = []

        if "input_shape" in self.context and "expected_shape" in self.context:
            suggestions.append(
                f"Expected input shape {self.context['expected_shape']}, got {self.context['input_shape']}"
            )

        if "parameter" in self.context and "value" in self.context:
            suggestions.append(
                f"Parameter '{self.context['parameter']}' has invalid value: {self.context['value']}"
            )

        if "expected" in self.context:
            suggestions.append(f"Expected: {self.context['expected']}")

        if suggestions:
            return f"{base_message}\n\nDetails:\n" + "\n".join(
                f"  • {s}" for s in suggestions
            )

        return base_message


class ConfigurationError(CReSOError):
    """Raised when there's an error in model configuration.

    Examples:
        - Invalid parameter values
        - Incompatible parameter combinations
        - Missing required parameters
    """

    pass


class ValidationError(CReSOError):
    """Raised when input validation fails.

    Examples:
        - Invalid input shapes
        - Out-of-range parameter values
        - Incompatible data types
    """

    pass


class TrainingError(CReSOError):
    """Raised when training encounters an error.

    Examples:
        - Training convergence failures
        - Gradient computation errors
        - Optimizer failures
    """

    pass


class ModelError(CReSOError):
    """Raised when model operations fail.

    Examples:
        - Forward pass failures
        - Model loading/saving errors
        - Export failures
    """

    pass


class DataError(CReSOError):
    """Raised when data processing encounters an error.

    Examples:
        - Invalid data formats
        - Missing required data
        - Data transformation failures
    """

    pass


class AdapterError(CReSOError):
    """Raised when data adapters encounter errors.

    Examples:
        - Time-series windowing failures
        - Graph processing errors
        - Feature extraction failures
    """

    pass


class ExportError(CReSOError):
    """Raised when model export operations fail.

    Examples:
        - TorchScript export failures
        - ONNX export errors
        - Format compatibility issues
    """

    pass


class CompatibilityError(CReSOError):
    """Raised when compatibility checks fail.

    Examples:
        - PyTorch version incompatibility
        - CUDA availability issues
        - Dependency version conflicts
    """

    pass


# Convenience functions for raising common exceptions


def raise_configuration_error(
    message: str,
    parameter: Optional[str] = None,
    value: Optional[Any] = None,
    expected: Optional[str] = None,
) -> None:
    """Raise a configuration error with structured context.

    Args:
        message: Error description
        parameter: Parameter name that caused the error
        value: Invalid parameter value
        expected: Expected value or range
    """
    context: Dict[str, Any] = {}
    if parameter:
        context["parameter"] = parameter
    if value is not None:
        context["value"] = value
    if expected:
        context["expected"] = expected

    raise ConfigurationError(message, context=context)


def raise_validation_error(
    message: str,
    input_name: Optional[str] = None,
    input_shape: Optional[tuple] = None,
    expected_shape: Optional[tuple] = None,
    input_type: Optional[type] = None,
    expected_type: Optional[type] = None,
) -> None:
    """Raise a validation error with structured context.

    Args:
        message: Error description
        input_name: Name of the problematic input
        input_shape: Actual input shape
        expected_shape: Expected input shape
        input_type: Actual input type
        expected_type: Expected input type
    """
    context: Dict[str, Any] = {}
    if input_name:
        context["input_name"] = input_name
    if input_shape:
        context["input_shape"] = input_shape
    if expected_shape:
        context["expected_shape"] = expected_shape
    if input_type:
        context["input_type"] = input_type.__name__
    if expected_type:
        context["expected_type"] = expected_type.__name__

    raise ValidationError(message, context=context)


def raise_training_error(
    message: str,
    epoch: Optional[int] = None,
    batch: Optional[int] = None,
    loss_value: Optional[float] = None,
    metric_values: Optional[Dict[str, float]] = None,
) -> None:
    """Raise a training error with structured context.

    Args:
        message: Error description
        epoch: Current epoch number
        batch: Current batch number
        loss_value: Current loss value
        metric_values: Current metric values
    """
    context: Dict[str, Any] = {}
    if epoch is not None:
        context["epoch"] = epoch
    if batch is not None:
        context["batch"] = batch
    if loss_value is not None:
        context["loss_value"] = loss_value
    if metric_values:
        context.update(metric_values)

    raise TrainingError(message, context=context)


def raise_model_error(
    message: str,
    model_class: Optional[str] = None,
    operation: Optional[str] = None,
    input_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Raise a model error with structured context.

    Args:
        message: Error description
        model_class: Model class name
        operation: Operation that failed
        input_info: Information about inputs
    """
    context: Dict[str, Any] = {}
    if model_class:
        context["model_class"] = model_class
    if operation:
        context["operation"] = operation
    if input_info:
        context.update(input_info)

    raise ModelError(message, context=context)


def raise_data_error(
    message: str,
    data_type: Optional[str] = None,
    data_shape: Optional[tuple] = None,
    data_source: Optional[str] = None,
) -> None:
    """Raise a data error with structured context.

    Args:
        message: Error description
        data_type: Type of data that caused the error
        data_shape: Shape of problematic data
        data_source: Source of the data
    """
    context: Dict[str, Any] = {}
    if data_type:
        context["data_type"] = data_type
    if data_shape:
        context["data_shape"] = data_shape
    if data_source:
        context["data_source"] = data_source

    raise DataError(message, context=context)
