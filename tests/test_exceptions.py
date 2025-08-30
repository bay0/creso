"""Test suite for creso.exceptions module."""

import pytest
from creso.exceptions import (
    CReSOError,
    ConfigurationError,
    ValidationError,
    TrainingError,
    ModelError,
    DataError,
    AdapterError,
    ExportError,
    CompatibilityError,
    raise_configuration_error,
    raise_validation_error,
    raise_training_error,
    raise_model_error,
    raise_data_error,
)


class TestCReSOError:
    """Test base CReSOError class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = CReSOError("Something went wrong")

        assert str(error) == "CReSOError: Something went wrong"
        assert error.message == "Something went wrong"
        assert error.error_code == "CReSOError"
        assert error.context == {}

    def test_error_with_code(self):
        """Test error with custom error code."""
        error = CReSOError("Test message", error_code="CUSTOM_CODE")

        assert error.error_code == "CUSTOM_CODE"
        assert "CUSTOM_CODE: Test message" in str(error)

    def test_error_with_context(self):
        """Test error with context information."""
        context = {"param": "value", "number": 42}
        error = CReSOError("Test message", context=context)

        assert error.context == context
        error_str = str(error)
        assert "param=value" in error_str
        assert "number=42" in error_str

    def test_error_repr(self):
        """Test error representation."""
        error = CReSOError("Test message", "CODE", {"key": "value"})
        repr_str = repr(error)

        assert "CReSOError" in repr_str
        assert "Test message" in repr_str
        assert "CODE" in repr_str
        assert "key" in repr_str

    def test_to_dict(self):
        """Test conversion to dictionary."""
        error = CReSOError("Test message", "CODE", {"key": "value"})
        error_dict = error.to_dict()

        expected = {
            "error_class": "CReSOError",
            "error_code": "CODE",
            "message": "Test message",
            "context": {"key": "value"},
        }
        assert error_dict == expected

    def test_get_user_friendly_message_basic(self):
        """Test basic user-friendly message."""
        error = CReSOError("Test message")
        friendly = error.get_user_friendly_message()

        assert friendly == "Test message"

    def test_get_user_friendly_message_with_shapes(self):
        """Test user-friendly message with shape information."""
        context = {"input_shape": (10, 5), "expected_shape": (10, 3)}
        error = CReSOError("Shape mismatch", context=context)
        friendly = error.get_user_friendly_message()

        assert "Expected input shape (10, 3), got (10, 5)" in friendly
        assert "Details:" in friendly

    def test_get_user_friendly_message_with_parameter(self):
        """Test user-friendly message with parameter information."""
        context = {"parameter": "learning_rate", "value": -0.1}
        error = CReSOError("Invalid parameter", context=context)
        friendly = error.get_user_friendly_message()

        assert "Parameter 'learning_rate' has invalid value: -0.1" in friendly

    def test_get_user_friendly_message_with_expected(self):
        """Test user-friendly message with expected value."""
        context = {"expected": "positive number"}
        error = CReSOError("Invalid value", context=context)
        friendly = error.get_user_friendly_message()

        assert "Expected: positive number" in friendly

    def test_get_user_friendly_message_multiple_details(self):
        """Test user-friendly message with multiple detail types."""
        context = {
            "input_shape": (10, 5),
            "expected_shape": (10, 3),
            "parameter": "input_dim",
            "value": 5,
            "expected": "3",
        }
        error = CReSOError("Multiple issues", context=context)
        friendly = error.get_user_friendly_message()

        assert "Expected input shape" in friendly
        assert "Parameter 'input_dim'" in friendly
        assert "Expected: 3" in friendly
        assert friendly.count("•") == 3  # Three bullet points


class TestSpecificErrors:
    """Test specific error classes."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid config")
        assert isinstance(error, CReSOError)
        assert error.error_code == "ConfigurationError"

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid input")
        assert isinstance(error, CReSOError)
        assert error.error_code == "ValidationError"

    def test_training_error(self):
        """Test TrainingError."""
        error = TrainingError("Training failed")
        assert isinstance(error, CReSOError)
        assert error.error_code == "TrainingError"

    def test_model_error(self):
        """Test ModelError."""
        error = ModelError("Model failed")
        assert isinstance(error, CReSOError)
        assert error.error_code == "ModelError"

    def test_data_error(self):
        """Test DataError."""
        error = DataError("Data processing failed")
        assert isinstance(error, CReSOError)
        assert error.error_code == "DataError"

    def test_adapter_error(self):
        """Test AdapterError."""
        error = AdapterError("Adapter failed")
        assert isinstance(error, CReSOError)
        assert error.error_code == "AdapterError"

    def test_export_error(self):
        """Test ExportError."""
        error = ExportError("Export failed")
        assert isinstance(error, CReSOError)
        assert error.error_code == "ExportError"

    def test_compatibility_error(self):
        """Test CompatibilityError."""
        error = CompatibilityError("Compatibility issue")
        assert isinstance(error, CReSOError)
        assert error.error_code == "CompatibilityError"


class TestConvenienceFunctions:
    """Test convenience functions for raising errors."""

    def test_raise_configuration_error_basic(self):
        """Test basic configuration error raising."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise_configuration_error("Invalid parameter")

        error = exc_info.value
        assert error.message == "Invalid parameter"
        assert error.context == {}

    def test_raise_configuration_error_with_context(self):
        """Test configuration error with context."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise_configuration_error(
                "Invalid parameter",
                parameter="learning_rate",
                value=-0.1,
                expected="positive number",
            )

        error = exc_info.value
        assert error.context["parameter"] == "learning_rate"
        assert error.context["value"] == -0.1
        assert error.context["expected"] == "positive number"

    def test_raise_validation_error_basic(self):
        """Test basic validation error raising."""
        with pytest.raises(ValidationError) as exc_info:
            raise_validation_error("Invalid input")

        error = exc_info.value
        assert error.message == "Invalid input"

    def test_raise_validation_error_with_shapes(self):
        """Test validation error with shape information."""
        with pytest.raises(ValidationError) as exc_info:
            raise_validation_error(
                "Shape mismatch",
                input_name="X",
                input_shape=(10, 5),
                expected_shape=(10, 3),
            )

        error = exc_info.value
        assert error.context["input_name"] == "X"
        assert error.context["input_shape"] == (10, 5)
        assert error.context["expected_shape"] == (10, 3)

    def test_raise_validation_error_with_types(self):
        """Test validation error with type information."""
        with pytest.raises(ValidationError) as exc_info:
            raise_validation_error("Type mismatch", input_type=str, expected_type=int)

        error = exc_info.value
        assert error.context["input_type"] == "str"
        assert error.context["expected_type"] == "int"

    def test_raise_training_error_basic(self):
        """Test basic training error raising."""
        with pytest.raises(TrainingError) as exc_info:
            raise_training_error("Training failed")

        error = exc_info.value
        assert error.message == "Training failed"

    def test_raise_training_error_with_context(self):
        """Test training error with training context."""
        metrics = {"accuracy": 0.5, "loss": 1.2}

        with pytest.raises(TrainingError) as exc_info:
            raise_training_error(
                "Training failed",
                epoch=10,
                batch=100,
                loss_value=1.5,
                metric_values=metrics,
            )

        error = exc_info.value
        assert error.context["epoch"] == 10
        assert error.context["batch"] == 100
        assert error.context["loss_value"] == 1.5
        assert error.context["accuracy"] == 0.5
        assert error.context["loss"] == 1.2

    def test_raise_model_error_basic(self):
        """Test basic model error raising."""
        with pytest.raises(ModelError) as exc_info:
            raise_model_error("Model operation failed")

        error = exc_info.value
        assert error.message == "Model operation failed"

    def test_raise_model_error_with_context(self):
        """Test model error with model context."""
        input_info = {"shape": (32, 10), "dtype": "float32"}

        with pytest.raises(ModelError) as exc_info:
            raise_model_error(
                "Forward pass failed",
                model_class="CReSOModel",
                operation="forward",
                input_info=input_info,
            )

        error = exc_info.value
        assert error.context["model_class"] == "CReSOModel"
        assert error.context["operation"] == "forward"
        assert error.context["shape"] == (32, 10)
        assert error.context["dtype"] == "float32"

    def test_raise_data_error_basic(self):
        """Test basic data error raising."""
        with pytest.raises(DataError) as exc_info:
            raise_data_error("Data processing failed")

        error = exc_info.value
        assert error.message == "Data processing failed"

    def test_raise_data_error_with_context(self):
        """Test data error with data context."""
        with pytest.raises(DataError) as exc_info:
            raise_data_error(
                "Invalid data format",
                data_type="timeseries",
                data_shape=(100, 10),
                data_source="file.csv",
            )

        error = exc_info.value
        assert error.context["data_type"] == "timeseries"
        assert error.context["data_shape"] == (100, 10)
        assert error.context["data_source"] == "file.csv"


class TestErrorInheritance:
    """Test error inheritance and behavior."""

    def test_error_inheritance(self):
        """Test that all errors inherit from CReSOError."""
        errors = [
            ConfigurationError("test"),
            ValidationError("test"),
            TrainingError("test"),
            ModelError("test"),
            DataError("test"),
            AdapterError("test"),
            ExportError("test"),
            CompatibilityError("test"),
        ]

        for error in errors:
            assert isinstance(error, CReSOError)
            assert isinstance(error, Exception)

    def test_error_catching(self):
        """Test that specific errors can be caught as CReSOError."""
        try:
            raise ValidationError("test error")
        except CReSOError as e:
            assert isinstance(e, ValidationError)
            assert e.message == "test error"

    def test_error_context_preserved(self):
        """Test that context is preserved through inheritance."""
        context = {"key": "value"}
        error = ValidationError("test", "CODE", context)

        # All methods should work the same
        assert error.context == context
        assert "key=value" in str(error)
        assert error.to_dict()["context"] == context
