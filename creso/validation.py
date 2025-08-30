"""
Input validation utilities for CReSO package.
Provides comprehensive validation functions with descriptive error messages
for configuration parameters, model inputs, and data formats.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Any, List, Optional, Tuple, Union
from pathlib import Path
from .exceptions import raise_validation_error


def validate_positive_int(
    value: Any, name: str, minimum: int = 1, maximum: Optional[int] = None
) -> int:
    """Validate that a value is a positive integer within range.
    Args:
        value: Value to validate
        name: Parameter name for error messages
        minimum: Minimum allowed value (inclusive)
        maximum: Maximum allowed value (inclusive)
    Returns:
        Validated integer value
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, int):
        raise_validation_error(
            f"{name} must be an integer",
            input_name=name,
            input_type=type(value),
            expected_type=int,
        )
    # mypy now knows value is int
    int_value: int = value
    if int_value < minimum:
        raise_validation_error(
            f"{name} must be >= {minimum}",
            input_name=name,
        )
    if maximum is not None and int_value > maximum:
        raise_validation_error(
            f"{name} must be <= {maximum}",
            input_name=name,
        )
    return int_value


def validate_positive_float(
    value: Any,
    name: str,
    minimum: float = 0.0,
    maximum: Optional[float] = None,
    inclusive_min: bool = True,
) -> float:
    """Validate that a value is a positive float within range.
    Args:
        value: Value to validate
        name: Parameter name for error messages
        minimum: Minimum allowed value
        maximum: Maximum allowed value
        inclusive_min: Whether minimum is inclusive
    Returns:
        Validated float value
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise_validation_error(
            f"{name} must be a number",
            input_name=name,
            input_type=type(value),
            expected_type=float,
        )
    float_value: float = float(value)
    if inclusive_min and float_value < minimum:
        raise_validation_error(
            f"{name} must be >= {minimum}",
            input_name=name,
        )
    elif not inclusive_min and float_value <= minimum:
        raise_validation_error(
            f"{name} must be > {minimum}",
            input_name=name,
        )
    if maximum is not None and float_value > maximum:
        raise_validation_error(
            f"{name} must be <= {maximum}",
            input_name=name,
        )
    return float_value


def validate_probability(value: Any, name: str) -> float:
    """Validate that a value is a valid probability [0, 1].
    Args:
        value: Value to validate
        name: Parameter name for error messages
    Returns:
        Validated probability value
    Raises:
        ValidationError: If validation fails
    """
    return validate_positive_float(value, name, minimum=0.0, maximum=1.0)


def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    name: str,
    allow_batch_dim: bool = True,
) -> torch.Tensor:
    """Validate tensor shape with optional batch dimension.
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (excluding batch dim if allow_batch_dim=True)
        name: Tensor name for error messages
        allow_batch_dim: Whether to allow an additional batch dimension
    Returns:
        Validated tensor
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(tensor, torch.Tensor):
        raise_validation_error(
            f"{name} must be a torch.Tensor",
            input_name=name,
            input_type=type(tensor),
            expected_type=torch.Tensor,
        )
    actual_shape = tensor.shape
    if allow_batch_dim:
        # Handle expected shape with None for batch dimension
        if len(expected_shape) == 2 and expected_shape[0] is None:
            # Expected shape is (None, dim) - check only the feature dimension
            if len(actual_shape) != 2:
                raise_validation_error(
                    f"{name} must be 2D tensor",
                    input_name=name,
                    input_shape=actual_shape,
                    expected_shape=expected_shape,
                )
            if actual_shape[1] != expected_shape[1]:
                raise_validation_error(
                    f"{name} has incorrect feature dimension",
                    input_name=name,
                    input_shape=actual_shape,
                    expected_shape=expected_shape,
                )
        # Allow additional batch dimension
        elif len(actual_shape) == len(expected_shape) + 1:
            # Has batch dimension - check remaining dimensions
            if actual_shape[1:] != expected_shape:
                raise_validation_error(
                    f"{name} has incorrect shape (excluding batch dimension)",
                    input_name=name,
                    input_shape=actual_shape[1:],
                    expected_shape=expected_shape,
                )
        elif len(actual_shape) == len(expected_shape):
            # No batch dimension - check all dimensions (skip None comparisons)
            for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
                if expected is not None and actual != expected:
                    raise_validation_error(
                        f"{name} has incorrect shape",
                        input_name=name,
                        input_shape=actual_shape,
                        expected_shape=expected_shape,
                    )
        else:
            raise_validation_error(
                f"{name} has incorrect number of dimensions",
                input_name=name,
                input_shape=actual_shape,
                expected_shape=expected_shape,
            )
    else:
        # Exact shape match required
        if actual_shape != expected_shape:
            raise_validation_error(
                f"{name} has incorrect shape",
                input_name=name,
                input_shape=actual_shape,
                expected_shape=expected_shape,
            )
    return tensor


def validate_tensor_2d(
    tensor: torch.Tensor,
    name: str,
    min_samples: int = 1,
    min_features: int = 1,
    max_samples: Optional[int] = None,
    max_features: Optional[int] = None,
) -> torch.Tensor:
    """Validate that tensor is 2D with valid dimensions.
    Args:
        tensor: Tensor to validate
        name: Tensor name for error messages
        min_samples: Minimum number of samples
        min_features: Minimum number of features
        max_samples: Maximum number of samples
        max_features: Maximum number of features
    Returns:
        Validated tensor
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(tensor, torch.Tensor):
        raise_validation_error(
            f"{name} must be a torch.Tensor",
            input_name=name,
            input_type=type(tensor),
            expected_type=torch.Tensor,
        )
    if tensor.dim() != 2:
        raise_validation_error(
            f"{name} must be 2-dimensional",
            input_name=name,
            input_shape=tensor.shape,
            expected_shape=("N", "D"),
        )
    n_samples, n_features = tensor.shape
    if n_samples < min_samples:
        raise_validation_error(
            f"{name} must have at least {min_samples} samples",
            input_name=name,
        )
    if n_features < min_features:
        raise_validation_error(
            f"{name} must have at least {min_features} features",
            input_name=name,
        )
    if max_samples is not None and n_samples > max_samples:
        raise_validation_error(
            f"{name} cannot have more than {max_samples} samples",
            input_name=name,
        )
    if max_features is not None and n_features > max_features:
        raise_validation_error(
            f"{name} cannot have more than {max_features} features",
            input_name=name,
        )
    return tensor


def validate_labels(
    labels: Union[torch.Tensor, np.ndarray],
    name: str = "labels",
    n_samples: Optional[int] = None,
    binary_classification: bool = False,
) -> torch.Tensor:
    """Validate classification labels.
    Args:
        labels: Labels to validate
        name: Labels name for error messages
        n_samples: Expected number of samples
        binary_classification: Whether to enforce binary labels
    Returns:
        Validated labels as torch.Tensor
    Raises:
        ValidationError: If validation fails
    """
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    elif not isinstance(labels, torch.Tensor):
        raise_validation_error(
            f"{name} must be torch.Tensor or numpy.ndarray",
            input_name=name,
            input_type=type(labels),
        )
    if labels.dim() > 2:
        raise_validation_error(
            f"{name} must be 1D or 2D", input_name=name, input_shape=labels.shape
        )
    # Flatten if 2D with single column
    if labels.dim() == 2 and labels.size(1) == 1:
        labels = labels.squeeze(1)
    if n_samples is not None and len(labels) != n_samples:
        raise_validation_error(
            f"{name} length must match number of samples",
            input_name=name,
        )
    # Check for valid classification labels
    unique_labels = torch.unique(labels)
    if binary_classification:
        if len(unique_labels) > 2:
            raise_validation_error(
                f"{name} must have at most 2 unique values for binary classification",
                input_name=name,
            )
        # Check if labels are 0/1
        valid_binary = torch.all((labels == 0) | (labels == 1))
        if not valid_binary:
            raise_validation_error(
                f"{name} for binary classification must be 0 or 1",
                input_name=name,
            )
    # Check for non-negative integer labels
    if not torch.all(labels >= 0):
        raise_validation_error(
            f"{name} must be non-negative",
            input_name=name,
        )
    if not torch.all(labels == labels.long()):
        raise_validation_error(f"{name} must be integers", input_name=name)
    return labels.long()


def validate_file_path(
    path: Union[str, Path],
    name: str,
    must_exist: bool = False,
    must_be_file: bool = True,
    allowed_extensions: Optional[List[str]] = None,
) -> Path:
    """Validate file path.
    Args:
        path: Path to validate
        name: Path name for error messages
        must_exist: Whether path must exist
        must_be_file: Whether path must be a file (vs directory)
        allowed_extensions: List of allowed file extensions
    Returns:
        Validated Path object
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(path, (str, Path)):
        raise_validation_error(
            f"{name} must be a string or Path", input_name=name, input_type=type(path)
        )
    path = Path(path)
    if must_exist and not path.exists():
        raise_validation_error(f"{name} does not exist", input_name=name)
    if must_exist and must_be_file and not path.is_file():
        raise_validation_error(f"{name} must be a file", input_name=name)
    if allowed_extensions and path.suffix.lower() not in allowed_extensions:
        raise_validation_error(
            f"{name} must have one of these extensions: {allowed_extensions}",
            input_name=name,
        )
    return path


def validate_device(device: Union[str, torch.device]) -> torch.device:
    """Validate PyTorch device.
    Args:
        device: Device specification
    Returns:
        Validated torch.device
    Raises:
        ValidationError: If validation fails
    """
    if isinstance(device, str):
        try:
            device = torch.device(device)
        except RuntimeError:
            raise_validation_error(
                f"Invalid device specification: {device}",
                input_name="device",
            )
    elif not isinstance(device, torch.device):
        raise_validation_error(
            "device must be string or torch.device",
            input_name="device",
            input_type=type(device),
            expected_type=torch.device,
        )
    # Check CUDA availability if CUDA device requested
    # At this point, device is guaranteed to be torch.device
    if device.type == "cuda" and not torch.cuda.is_available():
        raise_validation_error(
            "CUDA requested but not available",
            input_name="device",
        )
    return device


def validate_choice(value: Any, choices: List[Any], name: str) -> Any:
    """Validate that value is in allowed choices.
    Args:
        value: Value to validate
        choices: List of allowed values
        name: Parameter name for error messages
    Returns:
        Validated value
    Raises:
        ValidationError: If validation fails
    """
    if value not in choices:
        raise_validation_error(
            f"{name} must be one of: {choices}",
            input_name=name,
        )
    return value
