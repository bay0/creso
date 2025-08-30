"""Test suite for creso.utils module."""

import pytest
import torch
import numpy as np
from unittest.mock import patch

from creso.utils import set_global_seed, as_tensor, Standardizer


class TestSetGlobalSeed:
    """Test global seed setting functionality."""

    def test_set_seed_basic(self):
        """Test basic seed setting."""
        set_global_seed(42)

        # Test reproducibility
        torch.manual_seed(42)
        val1 = torch.rand(1).item()

        set_global_seed(42)
        val2 = torch.rand(1).item()

        assert val1 == val2

    def test_set_seed_deterministic_true(self):
        """Test deterministic mode enabled."""
        set_global_seed(42, deterministic=True)

        # Check that backends are set for determinism
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_set_seed_deterministic_false(self):
        """Test deterministic mode disabled."""
        set_global_seed(42, deterministic=False)

        # Should not modify cudnn settings when deterministic=False
        # We can't easily test this without knowing the initial state
        # Just ensure it runs without error
        assert True

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.manual_seed")
    @patch("torch.cuda.manual_seed_all")
    def test_set_seed_cuda_available(
        self, mock_seed_all, mock_seed, mock_cuda_available
    ):
        """Test seed setting with CUDA available."""
        set_global_seed(42)

        # Verify CUDA seeding functions were called
        assert mock_seed.call_count >= 1
        assert mock_seed_all.call_count >= 1
        mock_seed.assert_called_with(42)
        mock_seed_all.assert_called_with(42)

    @patch("torch.cuda.is_available", return_value=False)
    def test_set_seed_cuda_unavailable(self, mock_cuda_available):
        """Test seed setting with CUDA unavailable."""
        # Should not raise any errors
        set_global_seed(42)
        assert True


class TestAsTensor:
    """Test tensor conversion functionality."""

    def test_as_tensor_from_numpy(self):
        """Test conversion from numpy array."""
        arr = np.array([1, 2, 3])
        tensor = as_tensor(arr)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert torch.equal(tensor, torch.tensor([1.0, 2.0, 3.0]))

    def test_as_tensor_from_list(self):
        """Test conversion from list."""
        lst = [1, 2, 3]
        tensor = as_tensor(lst)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert torch.equal(tensor, torch.tensor([1.0, 2.0, 3.0]))

    def test_as_tensor_from_tensor(self):
        """Test conversion from existing tensor."""
        original = torch.tensor([1, 2, 3], dtype=torch.int64)
        tensor = as_tensor(original)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert torch.equal(tensor, torch.tensor([1.0, 2.0, 3.0]))

    def test_as_tensor_with_device(self):
        """Test tensor conversion with device specification."""
        arr = np.array([1, 2, 3])
        tensor = as_tensor(arr, device="cpu")

        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float32

    def test_as_tensor_with_dtype(self):
        """Test tensor conversion with custom dtype."""
        arr = np.array([1, 2, 3])
        tensor = as_tensor(arr, dtype=torch.int64)

        assert tensor.dtype == torch.int64
        assert torch.equal(tensor, torch.tensor([1, 2, 3], dtype=torch.int64))

    def test_as_tensor_with_device_and_dtype(self):
        """Test tensor conversion with both device and dtype."""
        arr = np.array([1.5, 2.5, 3.5])
        tensor = as_tensor(arr, device="cpu", dtype=torch.float64)

        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float64


class TestStandardizer:
    """Test Standardizer class functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample 2D data for testing."""
        torch.manual_seed(42)
        return torch.randn(100, 5) * 2 + 1

    @pytest.fixture
    def fitted_standardizer(self, sample_data):
        """Create a fitted standardizer."""
        standardizer = Standardizer()
        standardizer.fit(sample_data)
        return standardizer

    def test_init(self):
        """Test Standardizer initialization."""
        standardizer = Standardizer()

        assert standardizer.mean is None
        assert standardizer.std is None
        assert standardizer.fitted is False

    def test_fit_basic(self, sample_data):
        """Test basic fitting functionality."""
        standardizer = Standardizer()
        result = standardizer.fit(sample_data)

        assert result is standardizer  # Should return self
        assert standardizer.fitted is True
        assert standardizer.mean is not None
        assert standardizer.std is not None
        assert standardizer.mean.shape == (5,)
        assert standardizer.std.shape == (5,)

    def test_fit_invalid_dimensions(self):
        """Test fit with invalid input dimensions."""
        standardizer = Standardizer()

        # Test 1D input
        with pytest.raises(ValueError, match="Expected 2D input, got 1D"):
            standardizer.fit(torch.randn(10))

        # Test 3D input
        with pytest.raises(ValueError, match="Expected 2D input, got 3D"):
            standardizer.fit(torch.randn(10, 5, 3))

    def test_fit_zero_std(self):
        """Test fit with zero standard deviation (constant features)."""
        # Create data with constant features
        data = torch.ones(100, 3) * 5  # All features are constant

        standardizer = Standardizer()
        standardizer.fit(data)

        # Should clamp std to minimum value
        assert torch.all(standardizer.std >= 1e-8)

    def test_transform_basic(self, fitted_standardizer, sample_data):
        """Test basic transform functionality."""
        transformed = fitted_standardizer.transform(sample_data)

        assert transformed.shape == sample_data.shape
        assert isinstance(transformed, torch.Tensor)

        # Check that transformation produces approximately zero mean and unit variance
        assert torch.allclose(transformed.mean(dim=0), torch.zeros(5), atol=1e-5)
        assert torch.allclose(
            transformed.std(dim=0, unbiased=False), torch.ones(5), atol=1e-5
        )

    def test_transform_not_fitted(self):
        """Test transform before fitting."""
        standardizer = Standardizer()
        data = torch.randn(10, 5)

        with pytest.raises(
            RuntimeError, match="Standardizer must be fitted before transform"
        ):
            standardizer.transform(data)

    def test_transform_buffers_none(self, sample_data):
        """Test transform when buffers are None (edge case)."""
        standardizer = Standardizer()
        standardizer.fitted = True  # Set fitted but don't actually fit

        with pytest.raises(
            RuntimeError, match="Standardizer buffers not properly loaded"
        ):
            standardizer.transform(sample_data)

    def test_fit_transform(self, sample_data):
        """Test fit_transform convenience method."""
        standardizer = Standardizer()
        transformed = standardizer.fit_transform(sample_data)

        assert standardizer.fitted is True
        assert transformed.shape == sample_data.shape
        assert torch.allclose(transformed.mean(dim=0), torch.zeros(5), atol=1e-5)

    def test_inverse_transform(self, fitted_standardizer, sample_data):
        """Test inverse transformation."""
        transformed = fitted_standardizer.transform(sample_data)
        reconstructed = fitted_standardizer.inverse_transform(transformed)

        assert torch.allclose(reconstructed, sample_data, atol=1e-6)

    def test_inverse_transform_not_fitted(self):
        """Test inverse_transform before fitting."""
        standardizer = Standardizer()
        data = torch.randn(10, 5)

        with pytest.raises(
            RuntimeError, match="Standardizer must be fitted before inverse_transform"
        ):
            standardizer.inverse_transform(data)

    def test_inverse_transform_buffers_none(self):
        """Test inverse_transform when buffers are None."""
        standardizer = Standardizer()
        standardizer.fitted = True  # Set fitted but don't actually fit
        data = torch.randn(10, 5)

        with pytest.raises(
            RuntimeError, match="Standardizer buffers not properly loaded"
        ):
            standardizer.inverse_transform(data)

    def test_state_dict(self, fitted_standardizer):
        """Test state_dict includes fitted status."""
        state = fitted_standardizer.state_dict()

        assert "fitted" in state
        assert state["fitted"] is True
        assert "mean" in state
        assert "std" in state

    def test_load_state_dict_complete(self, sample_data):
        """Test loading complete state dict."""
        # Create and fit standardizer
        standardizer1 = Standardizer()
        standardizer1.fit(sample_data)
        state = standardizer1.state_dict()

        # Verify state dict contains required keys
        assert "fitted" in state
        assert "mean" in state
        assert "std" in state

        # Load into new standardizer
        standardizer2 = Standardizer()
        standardizer2.load_state_dict(state)

        # Check that fitted status was loaded
        assert standardizer2.fitted is True

    def test_load_state_dict_no_fitted(self, sample_data):
        """Test loading state dict without fitted flag."""
        standardizer1 = Standardizer()
        standardizer1.fit(sample_data)
        state = standardizer1.state_dict()

        # Remove fitted flag
        state_no_fitted = {k: v for k, v in state.items() if k != "fitted"}

        standardizer2 = Standardizer()
        standardizer2.load_state_dict(state_no_fitted)

        assert standardizer2.fitted is False  # Should default to False

    def test_load_state_dict_partial(self):
        """Test loading incomplete state dict."""
        standardizer = Standardizer()

        # Load state dict with only mean
        partial_state = {"mean": torch.tensor([1.0, 2.0, 3.0]), "fitted": True}
        standardizer.load_state_dict(partial_state)

        assert standardizer.fitted is True
        assert torch.equal(standardizer.mean, torch.tensor([1.0, 2.0, 3.0]))

    def test_load_state_dict_empty(self):
        """Test loading empty state dict."""
        standardizer = Standardizer()
        standardizer.load_state_dict({})

        assert standardizer.fitted is False

    def test_standardizer_persistence(self, sample_data, tmp_path):
        """Test saving and loading standardizer state dict."""
        # Fit standardizer
        standardizer1 = Standardizer()
        standardizer1.fit(sample_data)

        # Save and load state dict
        state_path = tmp_path / "standardizer.pt"
        torch.save(standardizer1.state_dict(), state_path)

        # Verify file was created and can be loaded
        assert state_path.exists()
        state = torch.load(state_path, map_location="cpu")

        # Check state dict has expected structure
        assert isinstance(state, dict)
        assert "fitted" in state
        assert state["fitted"] is True
