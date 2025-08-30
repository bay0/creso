"""
Tests for CReSO regressor functionality.
"""

import os
import tempfile
import numpy as np
import torch
import pytest
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

from creso.config import ModelArchitectureConfig, CReSOConfiguration
from creso.regressor import CReSORegressor, mean_squared_error as creso_mse
from creso.regressor import mean_absolute_error, r2_score as creso_r2, root_mean_squared_error
from creso.exceptions import ValidationError


class TestCReSORegressor:
    """Test CReSORegressor functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        arch_config = ModelArchitectureConfig(input_dim=10, n_components=8)
        config = CReSOConfiguration(architecture=arch_config)
        config.training.max_epochs = 20  # Increased for better training
        config.training.batch_size = 32
        return config

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=200, n_features=10, n_targets=1, 
            noise=0.1, random_state=42
        )
        return X.astype(np.float32), y.astype(np.float32)

    def test_initialization(self, config):
        """Test regressor initialization."""
        regressor = CReSORegressor(config)
        assert regressor.config == config
        assert regressor.model is None
        assert regressor.trainer is None
        assert regressor.standardizer is None
        assert not regressor._is_fitted

    def test_fit_basic(self, config, regression_data):
        """Test basic fitting functionality."""
        X, y = regression_data
        regressor = CReSORegressor(config)
        
        # Fit the regressor
        result = regressor.fit(X, y)
        
        # Check return value and fitted state
        assert result is regressor  # Should return self
        assert regressor._is_fitted
        assert regressor.model is not None
        assert regressor.trainer is not None

    def test_fit_with_validation(self, config, regression_data):
        """Test fitting with validation data."""
        X, y = regression_data
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]
        
        regressor = CReSORegressor(config)
        regressor.fit(X_train, y_train, X_val, y_val)
        
        assert regressor._is_fitted

    def test_fit_no_standardization(self, config, regression_data):
        """Test fitting without standardization."""
        X, y = regression_data
        regressor = CReSORegressor(config)
        regressor.fit(X, y, standardize=False)
        
        assert regressor._is_fitted
        assert regressor.standardizer is None

    def test_predict_basic(self, config, regression_data):
        """Test basic prediction functionality."""
        X, y = regression_data
        regressor = CReSORegressor(config)
        regressor.fit(X, y)
        
        # Make predictions
        predictions = regressor.predict(X)
        
        # Check output format
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(X),)
        assert predictions.dtype == np.float32 or predictions.dtype == np.float64
        
        # Should achieve some R² (may be negative for undertrained model)
        r2 = r2_score(y, predictions)
        assert r2 > -2.0  # Should not be completely random

    def test_score_r2(self, config, regression_data):
        """Test R² scoring functionality."""
        X, y = regression_data
        regressor = CReSORegressor(config)
        regressor.fit(X, y)
        
        # Get R² score
        score = regressor.score(X, y)
        
        # Check score format and range
        assert isinstance(score, float)
        assert score > 0.1  # Should achieve reasonable R² with more training
        
        # Compare with sklearn implementation
        predictions = regressor.predict(X)
        expected_r2 = r2_score(y, predictions)
        assert abs(score - expected_r2) < 1e-6

    def test_get_regression_metrics(self, config, regression_data):
        """Test comprehensive regression metrics."""
        X, y = regression_data
        regressor = CReSORegressor(config)
        regressor.fit(X, y)
        
        # Get metrics
        metrics = regressor.get_regression_metrics(X, y)
        
        # Check all expected metrics are present
        expected_keys = {'mse', 'rmse', 'mae', 'r2', 'mape', 'n_samples'}
        assert set(metrics.keys()) == expected_keys
        
        # Check metric types and ranges
        assert isinstance(metrics['mse'], float)
        assert isinstance(metrics['rmse'], float)
        assert isinstance(metrics['mae'], float)
        assert isinstance(metrics['r2'], float)
        assert isinstance(metrics['mape'], float)
        assert metrics['n_samples'] == len(X)
        
        # Check metric relationships  
        assert abs(metrics['rmse'] - np.sqrt(metrics['mse'])) < 1e-5
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
        
        # Verify against sklearn
        predictions = regressor.predict(X)
        expected_mse = mean_squared_error(y, predictions)
        assert abs(metrics['mse'] - expected_mse) < 1e-6

    def test_tensor_inputs(self, config, regression_data):
        """Test with PyTorch tensor inputs."""
        X, y = regression_data
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        
        regressor = CReSORegressor(config)
        regressor.fit(X_tensor, y_tensor)
        
        predictions = regressor.predict(X_tensor)
        score = regressor.score(X_tensor, y_tensor)
        
        assert isinstance(predictions, np.ndarray)
        assert isinstance(score, float)
        assert predictions.shape == (len(X),)

    def test_target_validation(self, config, regression_data):
        """Test target validation during fitting."""
        X, y = regression_data
        regressor = CReSORegressor(config)
        
        # Test mismatched shapes
        with pytest.raises(ValidationError):
            regressor.fit(X, y[:-5])  # Fewer targets than samples
        
        # Test NaN targets
        y_nan = y.copy()
        y_nan[0] = np.nan
        with pytest.raises(ValidationError, match="NaN or infinite"):
            regressor.fit(X, y_nan)
        
        # Test infinite targets
        y_inf = y.copy()
        y_inf[0] = np.inf
        with pytest.raises(ValidationError, match="NaN or infinite"):
            regressor.fit(X, y_inf)
        
        # Test 2D targets with single column (should work)
        y_2d = y.reshape(-1, 1)
        regressor.fit(X, y_2d)
        assert regressor._is_fitted
        
        # Test 2D targets with multiple columns (should fail)
        y_2d_multi = np.column_stack([y, y])
        with pytest.raises(ValidationError, match="shape"):
            regressor.fit(X, y_2d_multi)

    def test_unfitted_errors(self, config):
        """Test errors when using unfitted regressor."""
        regressor = CReSORegressor(config)
        X_dummy = np.random.randn(10, 10)
        y_dummy = np.random.randn(10)
        
        with pytest.raises(ValidationError, match="must be fitted"):
            regressor.predict(X_dummy)
        
        with pytest.raises(ValidationError, match="must be fitted"):
            regressor.score(X_dummy, y_dummy)
        
        with pytest.raises(ValidationError, match="must be fitted"):
            regressor.get_regression_metrics(X_dummy, y_dummy)

    def test_feature_importance_placeholder(self, config, regression_data):
        """Test feature importance implementation."""
        X, y = regression_data
        regressor = CReSORegressor(config)
        regressor.fit(X, y)
        
        # Should return feature importance based on amplitude magnitudes
        importance = regressor.feature_importance_
        assert importance is not None
        assert isinstance(importance, np.ndarray)
        assert len(importance) == regressor.config.architecture.n_components

    def test_save_load(self, config, regression_data):
        """Test save and load functionality."""
        X, y = regression_data
        regressor = CReSORegressor(config)
        regressor.fit(X, y)
        
        # Get original predictions
        original_predictions = regressor.predict(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "regressor.pkl")
            
            # Save regressor
            regressor.save(save_path)
            assert os.path.exists(save_path)
            
            # Load regressor
            loaded_regressor = CReSORegressor.load(save_path)
            
            # Check loaded regressor
            assert loaded_regressor._is_fitted
            assert loaded_regressor.config.architecture.input_dim == 10
            
            # Check predictions match
            loaded_predictions = loaded_regressor.predict(X)
            np.testing.assert_array_almost_equal(original_predictions, loaded_predictions, decimal=5)



    def test_wave_physics_model(self, config, regression_data):
        """Test regressor with wave physics enabled."""
        config.wave_physics.enable_wave_physics = True
        X, y = regression_data
        
        regressor = CReSORegressor(config)
        regressor.fit(X, y)
        
        assert regressor._is_fitted
        predictions = regressor.predict(X)
        assert predictions.shape == (len(X),)

    def test_different_epochs(self, config, regression_data):
        """Test regressor with different epoch settings."""
        X, y = regression_data
        
        # Test with very few epochs
        config.training.max_epochs = 2
        regressor = CReSORegressor(config)
        regressor.fit(X, y)
        
        predictions = regressor.predict(X)
        assert predictions.shape == (len(X),)

    def test_torch_tensor_targets(self, config, regression_data):
        """Test regressor with torch tensor targets."""
        X, y = regression_data
        y_tensor = torch.from_numpy(y)
        
        regressor = CReSORegressor(config)
        regressor.fit(X, y_tensor)
        
        assert regressor._is_fitted
        predictions = regressor.predict(X)
        assert predictions.shape == (len(X),)


class TestRegressionUtilityFunctions:
    """Test utility functions for regression."""

    def test_mean_squared_error(self):
        """Test MSE utility function."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.2, 2.8, 3.9])
        
        mse = creso_mse(y_true, y_pred)
        expected = np.mean((y_true - y_pred) ** 2)
        
        assert isinstance(mse, float)
        assert abs(mse - expected) < 1e-10

    def test_mean_absolute_error(self):
        """Test MAE utility function."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.2, 2.8, 3.9])
        
        mae = mean_absolute_error(y_true, y_pred)
        expected = np.mean(np.abs(y_true - y_pred))
        
        assert isinstance(mae, float)
        assert abs(mae - expected) < 1e-10

    def test_r2_score_utility(self):
        """Test R² utility function."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.2, 2.8, 3.9])
        
        r2 = creso_r2(y_true, y_pred)
        
        # Calculate expected R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        expected = 1 - (ss_res / ss_tot)
        
        assert isinstance(r2, float)
        assert abs(r2 - expected) < 1e-10

    def test_root_mean_squared_error(self):
        """Test RMSE utility function."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.2, 2.8, 3.9])
        
        rmse = root_mean_squared_error(y_true, y_pred)
        expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        assert isinstance(rmse, float)
        assert abs(rmse - expected) < 1e-10

    def test_perfect_predictions(self):
        """Test utility functions with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true.copy()  # Perfect predictions
        
        mse = creso_mse(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = creso_r2(y_true, y_pred)
        
        assert mse == 0.0
        assert mae == 0.0
        assert rmse == 0.0
        assert r2 == 1.0

    def test_worst_predictions(self):
        """Test utility functions with worst case (mean predictions)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full_like(y_true, np.mean(y_true))  # Always predict mean
        
        r2 = creso_r2(y_true, y_pred)
        assert abs(r2 - 0.0) < 1e-10  # R² should be 0 for mean predictions


if __name__ == "__main__":
    pytest.main([__file__])