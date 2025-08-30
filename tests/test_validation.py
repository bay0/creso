"""Test suite for creso.validation module."""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch

from creso.validation import (
    validate_positive_int,
    validate_positive_float,
    validate_probability,
    validate_tensor_shape,
    validate_tensor_2d,
    validate_labels,
    validate_file_path,
    validate_device,
    validate_choice,
)
from creso.exceptions import ValidationError


class TestValidatePositiveInt:
    """Test positive integer validation."""

    def test_valid_positive_int(self):
        """Test valid positive integer."""
        result = validate_positive_int(5, "test_param")
        assert result == 5

    def test_valid_int_with_custom_minimum(self):
        """Test valid integer with custom minimum."""
        result = validate_positive_int(10, "test_param", minimum=5)
        assert result == 10

    def test_valid_int_with_maximum(self):
        """Test valid integer with maximum constraint."""
        result = validate_positive_int(5, "test_param", minimum=1, maximum=10)
        assert result == 5

    def test_invalid_type_string(self):
        """Test invalid type (string)."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_int("5", "test_param")

        error = exc_info.value
        assert "must be an integer" in error.message
        assert error.context["input_name"] == "test_param"

    def test_invalid_type_float(self):
        """Test invalid type (float)."""
        with pytest.raises(ValidationError):
            validate_positive_int(5.5, "test_param")

    def test_below_minimum_default(self):
        """Test value below default minimum (1)."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_int(0, "test_param")

        assert "must be >= 1" in exc_info.value.message

    def test_below_custom_minimum(self):
        """Test value below custom minimum."""
        with pytest.raises(ValidationError):
            validate_positive_int(3, "test_param", minimum=5)

    def test_above_maximum(self):
        """Test value above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_int(15, "test_param", maximum=10)

        assert "must be <= 10" in exc_info.value.message

    def test_edge_cases(self):
        """Test edge cases."""
        # Exactly at minimum
        assert validate_positive_int(5, "test", minimum=5) == 5

        # Exactly at maximum
        assert validate_positive_int(10, "test", maximum=10) == 10


class TestValidatePositiveFloat:
    """Test positive float validation."""

    def test_valid_positive_float(self):
        """Test valid positive float."""
        result = validate_positive_float(3.14, "test_param")
        assert result == 3.14

    def test_valid_int_converted_to_float(self):
        """Test valid integer converted to float."""
        result = validate_positive_float(5, "test_param")
        assert result == 5.0
        assert isinstance(result, float)

    def test_valid_float_with_custom_minimum(self):
        """Test valid float with custom minimum."""
        result = validate_positive_float(2.5, "test_param", minimum=1.0)
        assert result == 2.5

    def test_valid_float_with_maximum(self):
        """Test valid float with maximum constraint."""
        result = validate_positive_float(7.5, "test_param", minimum=0.0, maximum=10.0)
        assert result == 7.5

    def test_exclusive_minimum(self):
        """Test exclusive minimum constraint."""
        result = validate_positive_float(
            0.1, "test_param", minimum=0.0, inclusive_min=False
        )
        assert result == 0.1

    def test_invalid_type(self):
        """Test invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_float("3.14", "test_param")

        assert "must be a number" in exc_info.value.message

    def test_below_inclusive_minimum(self):
        """Test value below inclusive minimum."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_float(-1.0, "test_param", minimum=0.0)

        assert "must be >= 0.0" in exc_info.value.message

    def test_below_exclusive_minimum(self):
        """Test value at exclusive minimum."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_float(0.0, "test_param", minimum=0.0, inclusive_min=False)

        assert "must be > 0.0" in exc_info.value.message

    def test_above_maximum(self):
        """Test value above maximum."""
        with pytest.raises(ValidationError):
            validate_positive_float(15.0, "test_param", maximum=10.0)

    def test_edge_cases(self):
        """Test edge cases."""
        # Zero with inclusive minimum
        assert validate_positive_float(0.0, "test", minimum=0.0) == 0.0

        # Exactly at maximum
        assert validate_positive_float(1.0, "test", maximum=1.0) == 1.0


class TestValidateProbability:
    """Test probability validation."""

    def test_valid_probabilities(self):
        """Test valid probability values."""
        assert validate_probability(0.0, "prob") == 0.0
        assert validate_probability(0.5, "prob") == 0.5
        assert validate_probability(1.0, "prob") == 1.0

    def test_invalid_negative_probability(self):
        """Test invalid negative probability."""
        with pytest.raises(ValidationError):
            validate_probability(-0.1, "prob")

    def test_invalid_probability_above_one(self):
        """Test invalid probability above 1."""
        with pytest.raises(ValidationError):
            validate_probability(1.1, "prob")


class TestValidateTensorShape:
    """Test tensor shape validation."""

    def test_valid_tensor_exact_shape(self):
        """Test valid tensor with exact shape match."""
        tensor = torch.randn(10, 5)
        result = validate_tensor_shape(
            tensor, (10, 5), "test_tensor", allow_batch_dim=False
        )

        assert torch.equal(result, tensor)

    def test_valid_tensor_with_batch_dim(self):
        """Test valid tensor with batch dimension."""
        tensor = torch.randn(32, 10, 5)
        result = validate_tensor_shape(
            tensor, (10, 5), "test_tensor", allow_batch_dim=True
        )

        assert torch.equal(result, tensor)

    def test_valid_tensor_no_batch_dim(self):
        """Test valid tensor without batch dimension."""
        tensor = torch.randn(10, 5)
        result = validate_tensor_shape(
            tensor, (10, 5), "test_tensor", allow_batch_dim=True
        )

        assert torch.equal(result, tensor)

    def test_valid_tensor_with_none_batch_dim(self):
        """Test valid tensor with None as batch dimension."""
        tensor = torch.randn(32, 10)
        result = validate_tensor_shape(
            tensor, (None, 10), "test_tensor", allow_batch_dim=True
        )

        assert torch.equal(result, tensor)

    def test_invalid_type(self):
        """Test invalid input type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_tensor_shape([1, 2, 3], (3,), "test_tensor")

        assert "must be a torch.Tensor" in exc_info.value.message

    def test_invalid_shape_exact(self):
        """Test invalid shape with exact matching."""
        tensor = torch.randn(10, 3)

        with pytest.raises(ValidationError) as exc_info:
            validate_tensor_shape(tensor, (10, 5), "test_tensor", allow_batch_dim=False)

        error = exc_info.value
        assert error.context["input_shape"] == (10, 3)
        assert error.context["expected_shape"] == (10, 5)

    def test_invalid_feature_dim_with_none_batch(self):
        """Test invalid feature dimension with None batch dimension."""
        tensor = torch.randn(32, 5)

        with pytest.raises(ValidationError) as exc_info:
            validate_tensor_shape(tensor, (None, 10), "test_tensor")

        assert "incorrect feature dimension" in exc_info.value.message

    def test_invalid_dimensions_with_batch(self):
        """Test invalid number of dimensions with batch allowance."""
        tensor = torch.randn(10, 5, 3, 2)  # 4D tensor

        with pytest.raises(ValidationError) as exc_info:
            validate_tensor_shape(tensor, (5, 3), "test_tensor", allow_batch_dim=True)

        assert "incorrect number of dimensions" in exc_info.value.message

    def test_invalid_not_2d_with_none_batch(self):
        """Test invalid tensor (not 2D) when expecting (None, dim)."""
        tensor = torch.randn(10, 5, 3)

        with pytest.raises(ValidationError) as exc_info:
            validate_tensor_shape(tensor, (None, 5), "test_tensor")

        assert "must be 2D tensor" in exc_info.value.message


class TestValidateTensor2D:
    """Test 2D tensor validation."""

    def test_valid_2d_tensor(self):
        """Test valid 2D tensor."""
        tensor = torch.randn(100, 10)
        result = validate_tensor_2d(tensor, "test_tensor")

        assert torch.equal(result, tensor)

    def test_valid_2d_tensor_with_constraints(self):
        """Test valid 2D tensor with size constraints."""
        tensor = torch.randn(50, 5)
        result = validate_tensor_2d(
            tensor,
            "test_tensor",
            min_samples=10,
            min_features=3,
            max_samples=100,
            max_features=10,
        )

        assert torch.equal(result, tensor)

    def test_invalid_type(self):
        """Test invalid input type."""
        with pytest.raises(ValidationError):
            validate_tensor_2d(np.array([[1, 2], [3, 4]]), "test_tensor")

    def test_invalid_1d_tensor(self):
        """Test invalid 1D tensor."""
        tensor = torch.randn(10)

        with pytest.raises(ValidationError) as exc_info:
            validate_tensor_2d(tensor, "test_tensor")

        assert "must be 2-dimensional" in exc_info.value.message

    def test_invalid_3d_tensor(self):
        """Test invalid 3D tensor."""
        tensor = torch.randn(10, 5, 3)

        with pytest.raises(ValidationError):
            validate_tensor_2d(tensor, "test_tensor")

    def test_insufficient_samples(self):
        """Test tensor with insufficient samples."""
        tensor = torch.randn(5, 10)

        with pytest.raises(ValidationError) as exc_info:
            validate_tensor_2d(tensor, "test_tensor", min_samples=10)

        assert "must have at least 10 samples" in exc_info.value.message

    def test_insufficient_features(self):
        """Test tensor with insufficient features."""
        tensor = torch.randn(100, 2)

        with pytest.raises(ValidationError) as exc_info:
            validate_tensor_2d(tensor, "test_tensor", min_features=5)

        assert "must have at least 5 features" in exc_info.value.message

    def test_too_many_samples(self):
        """Test tensor with too many samples."""
        tensor = torch.randn(150, 10)

        with pytest.raises(ValidationError) as exc_info:
            validate_tensor_2d(tensor, "test_tensor", max_samples=100)

        assert "cannot have more than 100 samples" in exc_info.value.message

    def test_too_many_features(self):
        """Test tensor with too many features."""
        tensor = torch.randn(100, 15)

        with pytest.raises(ValidationError) as exc_info:
            validate_tensor_2d(tensor, "test_tensor", max_features=10)

        assert "cannot have more than 10 features" in exc_info.value.message


class TestValidateLabels:
    """Test label validation."""

    def test_valid_torch_labels(self):
        """Test valid torch tensor labels."""
        labels = torch.tensor([0, 1, 2, 1, 0])
        result = validate_labels(labels)

        assert torch.equal(result, labels.long())

    def test_valid_numpy_labels(self):
        """Test valid numpy array labels."""
        labels = np.array([0, 1, 2, 1, 0])
        result = validate_labels(labels)

        assert torch.equal(result, torch.tensor([0, 1, 2, 1, 0], dtype=torch.long))

    def test_valid_binary_labels(self):
        """Test valid binary classification labels."""
        labels = torch.tensor([0, 1, 1, 0, 1])
        result = validate_labels(labels, binary_classification=True)

        assert torch.equal(result, labels.long())

    def test_valid_2d_labels_single_column(self):
        """Test valid 2D labels with single column."""
        labels = torch.tensor([[0], [1], [2], [1], [0]])
        result = validate_labels(labels)

        expected = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
        assert torch.equal(result, expected)

    def test_valid_labels_with_sample_count(self):
        """Test valid labels with sample count constraint."""
        labels = torch.tensor([0, 1, 2])
        result = validate_labels(labels, n_samples=3)

        assert torch.equal(result, labels.long())

    def test_invalid_type(self):
        """Test invalid label type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_labels([0, 1, 2, 1, 0])

        assert "must be torch.Tensor or numpy.ndarray" in exc_info.value.message

    def test_invalid_3d_labels(self):
        """Test invalid 3D labels."""
        labels = torch.randn(10, 5, 3)

        with pytest.raises(ValidationError) as exc_info:
            validate_labels(labels)

        assert "must be 1D or 2D" in exc_info.value.message

    def test_invalid_sample_count_mismatch(self):
        """Test sample count mismatch."""
        labels = torch.tensor([0, 1, 2])

        with pytest.raises(ValidationError) as exc_info:
            validate_labels(labels, n_samples=5)

        assert "length must match number of samples" in exc_info.value.message

    def test_invalid_too_many_binary_classes(self):
        """Test too many classes for binary classification."""
        labels = torch.tensor([0, 1, 2])

        with pytest.raises(ValidationError) as exc_info:
            validate_labels(labels, binary_classification=True)

        assert "at most 2 unique values" in exc_info.value.message

    def test_invalid_binary_label_values(self):
        """Test invalid binary label values (not 0/1)."""
        labels = torch.tensor([1, 2, 2, 1])

        with pytest.raises(ValidationError) as exc_info:
            validate_labels(labels, binary_classification=True)

        assert "must be 0 or 1" in exc_info.value.message

    def test_invalid_negative_labels(self):
        """Test invalid negative labels."""
        labels = torch.tensor([-1, 0, 1])

        with pytest.raises(ValidationError) as exc_info:
            validate_labels(labels)

        assert "must be non-negative" in exc_info.value.message

    def test_invalid_float_labels(self):
        """Test invalid float labels."""
        labels = torch.tensor([0.5, 1.5, 2.5])

        with pytest.raises(ValidationError) as exc_info:
            validate_labels(labels)

        assert "must be integers" in exc_info.value.message


class TestValidateFilePath:
    """Test file path validation."""

    def test_valid_path_string(self, tmp_path):
        """Test valid path as string."""
        result = validate_file_path(str(tmp_path), "test_path", must_exist=False)

        assert isinstance(result, Path)
        assert result == tmp_path

    def test_valid_path_object(self, tmp_path):
        """Test valid Path object."""
        result = validate_file_path(tmp_path, "test_path")

        assert result == tmp_path

    def test_existing_file(self, tmp_path):
        """Test existing file validation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = validate_file_path(test_file, "test_path", must_exist=True)

        assert result == test_file

    def test_valid_extension(self, tmp_path):
        """Test valid file extension."""
        test_file = tmp_path / "test.txt"

        result = validate_file_path(
            test_file,
            "test_path",
            must_exist=False,
            allowed_extensions=[".txt", ".csv"],
        )

        assert result == test_file

    def test_invalid_type(self):
        """Test invalid path type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_path(123, "test_path")

        assert "must be a string or Path" in exc_info.value.message

    def test_nonexistent_file_must_exist(self, tmp_path):
        """Test nonexistent file when must_exist=True."""
        nonexistent = tmp_path / "nonexistent.txt"

        with pytest.raises(ValidationError) as exc_info:
            validate_file_path(nonexistent, "test_path", must_exist=True)

        assert "does not exist" in exc_info.value.message

    def test_directory_when_file_required(self, tmp_path):
        """Test directory when file is required."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_path(
                tmp_path, "test_path", must_exist=True, must_be_file=True
            )

        assert "must be a file" in exc_info.value.message

    def test_invalid_extension(self, tmp_path):
        """Test invalid file extension."""
        test_file = tmp_path / "test.json"

        with pytest.raises(ValidationError) as exc_info:
            validate_file_path(
                test_file, "test_path", allowed_extensions=[".txt", ".csv"]
            )

        assert "must have one of these extensions" in exc_info.value.message


class TestValidateDevice:
    """Test device validation."""

    def test_valid_device_string_cpu(self):
        """Test valid CPU device string."""
        result = validate_device("cpu")

        assert isinstance(result, torch.device)
        assert result.type == "cpu"

    def test_valid_device_object(self):
        """Test valid device object."""
        device = torch.device("cpu")
        result = validate_device(device)

        assert result is device

    def test_invalid_device_string(self):
        """Test invalid device string."""
        with pytest.raises(ValidationError) as exc_info:
            validate_device("invalid_device")

        assert "Invalid device specification" in exc_info.value.message

    def test_invalid_device_type(self):
        """Test invalid device type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_device(123)

        error = exc_info.value
        assert "must be string or torch.device" in error.message
        assert error.context["expected_type"] == "device"

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_requested_but_unavailable(self, mock_cuda):
        """Test CUDA requested when not available."""
        with pytest.raises(ValidationError) as exc_info:
            validate_device("cuda")

        assert "CUDA requested but not available" in exc_info.value.message

    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_requested_and_available(self, mock_cuda):
        """Test CUDA requested when available."""
        result = validate_device("cuda")

        assert result.type == "cuda"


class TestValidateChoice:
    """Test choice validation."""

    def test_valid_choice(self):
        """Test valid choice from list."""
        choices = ["option1", "option2", "option3"]
        result = validate_choice("option2", choices, "test_param")

        assert result == "option2"

    def test_valid_choice_numeric(self):
        """Test valid numeric choice."""
        choices = [1, 2, 3, 5, 8]
        result = validate_choice(5, choices, "test_param")

        assert result == 5

    def test_invalid_choice(self):
        """Test invalid choice."""
        choices = ["option1", "option2", "option3"]

        with pytest.raises(ValidationError) as exc_info:
            validate_choice("invalid_option", choices, "test_param")

        error = exc_info.value
        assert "must be one of:" in error.message
        assert "option1" in error.message

    def test_empty_choices(self):
        """Test validation with empty choices list."""
        with pytest.raises(ValidationError):
            validate_choice("anything", [], "test_param")
