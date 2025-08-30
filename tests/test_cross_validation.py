"""
Tests for CReSO cross-validation utilities.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import StratifiedKFold

from creso.config import ModelArchitectureConfig, CReSOConfiguration
from creso.classifier import CReSOClassifier
from creso.regressor import CReSORegressor
from creso.cross_validation import (
    cross_val_score, cross_validate, validation_curve,
    _clone_estimator, _get_scorer, _set_nested_param
)
from creso.exceptions import ValidationError


class TestCrossValScore:
    """Test cross_val_score function."""

    @pytest.fixture
    def classifier_setup(self):
        """Create classifier and classification data."""
        arch_config = ModelArchitectureConfig(input_dim=10, n_components=6)
        config = CReSOConfiguration(architecture=arch_config)
        config.training.max_epochs = 3
        config.training.batch_size = 32
        
        classifier = CReSOClassifier(config)
        
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2,
            n_redundant=0, n_informative=8, random_state=42
        )
        
        return classifier, X.astype(np.float32), y.astype(np.int32)

    @pytest.fixture
    def regressor_setup(self):
        """Create regressor and regression data."""
        arch_config = ModelArchitectureConfig(input_dim=8, n_components=6)
        config = CReSOConfiguration(architecture=arch_config)
        config.training.max_epochs = 3
        config.training.batch_size = 32
        
        regressor = CReSORegressor(config)
        
        X, y = make_regression(
            n_samples=100, n_features=8, n_targets=1,
            noise=0.1, random_state=42
        )
        
        return regressor, X.astype(np.float32), y.astype(np.float32)

    def test_cross_val_score_classifier_default(self, classifier_setup):
        """Test cross_val_score with classifier using defaults."""
        classifier, X, y = classifier_setup
        
        scores = cross_val_score(classifier, X, y)
        
        # Check output format
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 5  # Default 5-fold CV
        assert all(0 <= score <= 1 for score in scores)  # Valid accuracy range

    def test_cross_val_score_classifier_custom_cv(self, classifier_setup):
        """Test cross_val_score with custom CV strategy."""
        classifier, X, y = classifier_setup
        
        # Test with integer CV
        scores = cross_val_score(classifier, X, y, cv=3)
        assert len(scores) == 3
        
        # Test with CV object
        cv_obj = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        scores = cross_val_score(classifier, X, y, cv=cv_obj)
        assert len(scores) == 4

    def test_cross_val_score_classifier_custom_scoring(self, classifier_setup):
        """Test cross_val_score with custom scoring."""
        classifier, X, y = classifier_setup
        
        # Test with accuracy scoring (explicit)
        scores = cross_val_score(classifier, X, y, scoring='accuracy')
        assert all(0 <= score <= 1 for score in scores)

    def test_cross_val_score_regressor_default(self, regressor_setup):
        """Test cross_val_score with regressor using defaults."""
        regressor, X, y = regressor_setup
        
        scores = cross_val_score(regressor, X, y)
        
        # Check output format
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 5  # Default 5-fold CV
        # R² scores can be negative for poor models, so just check they're reasonable
        assert all(-2 <= score <= 1 for score in scores)

    def test_cross_val_score_regressor_custom_scoring(self, regressor_setup):
        """Test cross_val_score with regressor custom scoring."""
        regressor, X, y = regressor_setup
        
        # Test with R² scoring (explicit)
        scores = cross_val_score(regressor, X, y, scoring='r2')
        assert all(-2 <= score <= 1 for score in scores)
        
        # Test with negative MSE scoring
        scores = cross_val_score(regressor, X, y, scoring='neg_mean_squared_error')
        assert all(score <= 0 for score in scores)  # Negative MSE should be <= 0
        
        # Test with negative MAE scoring
        scores = cross_val_score(regressor, X, y, scoring='neg_mean_absolute_error')
        assert all(score <= 0 for score in scores)  # Negative MAE should be <= 0

    def test_cross_val_score_verbose(self, classifier_setup):
        """Test cross_val_score with verbose output."""
        classifier, X, y = classifier_setup
        
        # Should not raise any errors with verbose=1
        scores = cross_val_score(classifier, X, y, verbose=1)
        assert len(scores) == 5

    def test_cross_val_score_fit_params(self, classifier_setup):
        """Test cross_val_score with fit parameters."""
        classifier, X, y = classifier_setup
        
        # Test with fit parameters
        fit_params = {'standardize': False}
        scores = cross_val_score(classifier, X, y, fit_params=fit_params)
        assert len(scores) == 5

    def test_cross_val_score_tensor_inputs(self, classifier_setup):
        """Test cross_val_score with tensor inputs."""
        import torch
        classifier, X, y = classifier_setup
        
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        
        scores = cross_val_score(classifier, X_tensor, y_tensor, cv=3)
        assert len(scores) == 3


class TestCrossValidate:
    """Test cross_validate function."""

    @pytest.fixture
    def classifier_setup(self):
        """Create classifier and classification data."""
        arch_config = ModelArchitectureConfig(input_dim=10, n_components=6)
        config = CReSOConfiguration(architecture=arch_config)
        config.training.max_epochs = 3
        
        classifier = CReSOClassifier(config)
        
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2,
            n_redundant=0, n_informative=8, random_state=42
        )
        
        return classifier, X.astype(np.float32), y.astype(np.int32)

    def test_cross_validate_basic(self, classifier_setup):
        """Test basic cross_validate functionality."""
        classifier, X, y = classifier_setup
        
        results = cross_validate(classifier, X, y, cv=3)
        
        # Check required keys
        assert 'test_accuracy' in results
        assert 'fit_time' in results
        assert 'score_time' in results
        
        # Check array shapes
        assert len(results['test_accuracy']) == 3
        assert len(results['fit_time']) == 3
        assert len(results['score_time']) == 3
        
        # Check that all are numpy arrays
        assert isinstance(results['test_accuracy'], np.ndarray)
        assert isinstance(results['fit_time'], np.ndarray)
        assert isinstance(results['score_time'], np.ndarray)

    def test_cross_validate_multiple_metrics(self, classifier_setup):
        """Test cross_validate with multiple metrics."""
        classifier, X, y = classifier_setup
        
        results = cross_validate(
            classifier, X, y, 
            scoring=['accuracy'], 
            cv=3
        )
        
        assert 'test_accuracy' in results
        assert len(results['test_accuracy']) == 3

    def test_cross_validate_return_train_score(self, classifier_setup):
        """Test cross_validate with return_train_score=True."""
        classifier, X, y = classifier_setup
        
        results = cross_validate(
            classifier, X, y, 
            cv=3, 
            return_train_score=True
        )
        
        assert 'test_accuracy' in results
        assert 'train_accuracy' in results
        assert len(results['train_accuracy']) == 3

    def test_cross_validate_return_estimator(self, classifier_setup):
        """Test cross_validate with return_estimator=True."""
        classifier, X, y = classifier_setup
        
        results = cross_validate(
            classifier, X, y, 
            cv=3, 
            return_estimator=True
        )
        
        assert 'estimator' in results
        assert len(results['estimator']) == 3
        
        # Check that estimators are fitted
        for estimator in results['estimator']:
            assert estimator._is_fitted

    def test_cross_validate_verbose(self, classifier_setup):
        """Test cross_validate with verbose output."""
        classifier, X, y = classifier_setup
        
        # Should not raise any errors with verbose=1
        results = cross_validate(classifier, X, y, cv=3, verbose=1)
        assert 'test_accuracy' in results

    def test_cross_validate_error_score(self, classifier_setup):
        """Test cross_validate with error handling."""
        classifier, X, y = classifier_setup
        
        # This should work normally
        results = cross_validate(
            classifier, X, y, 
            cv=3, 
            error_score=np.nan
        )
        
        assert 'test_accuracy' in results
        assert len(results['test_accuracy']) == 3


class TestValidationCurve:
    """Test validation_curve function."""

    @pytest.fixture
    def classifier_setup(self):
        """Create classifier and classification data."""
        arch_config = ModelArchitectureConfig(input_dim=8, n_components=4)
        config = CReSOConfiguration(architecture=arch_config)
        config.training.max_epochs = 2  # Very fast for testing
        
        classifier = CReSOClassifier(config)
        
        X, y = make_classification(
            n_samples=60, n_features=8, n_classes=2,
            n_redundant=0, n_informative=6, random_state=42
        )
        
        return classifier, X.astype(np.float32), y.astype(np.int32)

    def test_validation_curve_basic(self, classifier_setup):
        """Test basic validation_curve functionality."""
        classifier, X, y = classifier_setup
        
        # Test varying number of components
        param_range = [2, 4, 6]
        train_scores, val_scores = validation_curve(
            classifier, X, y,
            param_name='config.architecture.n_components',
            param_range=param_range,
            cv=3
        )
        
        # Check output shapes
        assert train_scores.shape == (len(param_range), 3)  # 3-fold CV
        assert val_scores.shape == (len(param_range), 3)
        
        # Check that we get one result per parameter value
        assert len(train_scores) == len(param_range)
        assert len(val_scores) == len(param_range)

    def test_validation_curve_verbose(self, classifier_setup):
        """Test validation_curve with verbose output."""
        classifier, X, y = classifier_setup
        
        param_range = [2, 4]
        train_scores, val_scores = validation_curve(
            classifier, X, y,
            param_name='config.architecture.n_components',
            param_range=param_range,
            cv=2,
            verbose=1
        )
        
        assert train_scores.shape == (2, 2)
        assert val_scores.shape == (2, 2)

    def test_validation_curve_custom_scoring(self, classifier_setup):
        """Test validation_curve with custom scoring."""
        classifier, X, y = classifier_setup
        
        param_range = [2, 4]
        train_scores, val_scores = validation_curve(
            classifier, X, y,
            param_name='config.architecture.n_components',
            param_range=param_range,
            cv=2,
            scoring='accuracy'
        )
        
        assert train_scores.shape == (2, 2)
        assert val_scores.shape == (2, 2)


class TestHelperFunctions:
    """Test cross-validation helper functions."""

    def test_clone_estimator_classifier(self):
        """Test _clone_estimator with classifier."""
        arch_config = ModelArchitectureConfig(input_dim=10, n_components=8)
        config = CReSOConfiguration(architecture=arch_config)
        classifier = CReSOClassifier(config)
        
        cloned = _clone_estimator(classifier)
        
        assert isinstance(cloned, CReSOClassifier)
        assert cloned.config == classifier.config
        assert cloned is not classifier  # Should be different object

    def test_clone_estimator_regressor(self):
        """Test _clone_estimator with regressor."""
        arch_config = ModelArchitectureConfig(input_dim=10, n_components=8)
        config = CReSOConfiguration(architecture=arch_config)
        regressor = CReSORegressor(config)
        
        cloned = _clone_estimator(regressor)
        
        assert isinstance(cloned, CReSORegressor)
        assert cloned.config == regressor.config
        assert cloned is not regressor

    def test_clone_estimator_unknown_type(self):
        """Test _clone_estimator with unknown estimator type."""
        fake_estimator = "not an estimator"
        
        with pytest.raises(ValidationError, match="Unknown estimator type"):
            _clone_estimator(fake_estimator)

    def test_get_scorer_accuracy(self):
        """Test _get_scorer for accuracy."""
        scorer = _get_scorer('accuracy')
        
        # Create a mock estimator with score method
        class MockEstimator:
            def score(self, X, y):
                return 0.85
        
        estimator = MockEstimator()
        X, y = np.random.randn(10, 5), np.random.randint(0, 2, 10)
        
        score = scorer(estimator, X, y)
        assert score == 0.85

    def test_get_scorer_r2(self):
        """Test _get_scorer for R²."""
        scorer = _get_scorer('r2')
        
        # Create a mock estimator with score method
        class MockEstimator:
            def score(self, X, y):
                return 0.75
        
        estimator = MockEstimator()
        X, y = np.random.randn(10, 5), np.random.randn(10)
        
        score = scorer(estimator, X, y)
        assert score == 0.75

    def test_get_scorer_neg_mse(self):
        """Test _get_scorer for negative MSE."""
        scorer = _get_scorer('neg_mean_squared_error')
        
        # Create a mock estimator with predict method
        class MockEstimator:
            def predict(self, X):
                return np.ones(len(X))
        
        estimator = MockEstimator()
        X = np.random.randn(10, 5)
        y = np.ones(10) * 2  # True values are 2, predictions are 1
        
        score = scorer(estimator, X, y)
        expected = -np.mean((y - np.ones(10)) ** 2)  # -(2-1)^2 = -1
        assert abs(score - expected) < 1e-10

    def test_get_scorer_neg_mae(self):
        """Test _get_scorer for negative MAE."""
        scorer = _get_scorer('neg_mean_absolute_error')
        
        # Create a mock estimator with predict method
        class MockEstimator:
            def predict(self, X):
                return np.ones(len(X))
        
        estimator = MockEstimator()
        X = np.random.randn(10, 5)
        y = np.ones(10) * 2  # True values are 2, predictions are 1
        
        score = scorer(estimator, X, y)
        expected = -np.mean(np.abs(y - np.ones(10)))  # -|2-1| = -1
        assert abs(score - expected) < 1e-10

    def test_get_scorer_unknown(self):
        """Test _get_scorer with unknown scoring metric."""
        with pytest.raises(ValidationError, match="Unknown scoring metric"):
            _get_scorer('unknown_metric')

    def test_set_nested_param_simple(self):
        """Test _set_nested_param with simple parameter."""
        class MockObject:
            def __init__(self):
                self.value = 10
        
        obj = MockObject()
        _set_nested_param(obj, 'value', 20)
        assert obj.value == 20

    def test_set_nested_param_nested(self):
        """Test _set_nested_param with nested parameter."""
        class MockInner:
            def __init__(self):
                self.param = 5
        
        class MockOuter:
            def __init__(self):
                self.inner = MockInner()
        
        obj = MockOuter()
        _set_nested_param(obj, 'inner.param', 15)
        assert obj.inner.param == 15

    def test_set_nested_param_deep_nested(self):
        """Test _set_nested_param with deeply nested parameter."""
        class MockConfig:
            def __init__(self):
                self.training = MockTraining()
        
        class MockTraining:
            def __init__(self):
                self.learning_rate = 0.01
        
        class MockEstimator:
            def __init__(self):
                self.config = MockConfig()
        
        estimator = MockEstimator()
        _set_nested_param(estimator, 'config.training.learning_rate', 0.001)
        assert estimator.config.training.learning_rate == 0.001


class TestCrossValidationEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_cv_parameter(self):
        """Test cross_val_score with invalid CV parameter."""
        arch_config = ModelArchitectureConfig(input_dim=10, n_components=6)
        config = CReSOConfiguration(architecture=arch_config)
        classifier = CReSOClassifier(config)
        
        X, y = make_classification(n_samples=50, n_features=10, random_state=42)
        
        # This should work - the function should handle various CV inputs
        scores = cross_val_score(classifier, X.astype(np.float32), y.astype(np.int32), cv=2)
        assert len(scores) == 2

    def test_small_dataset(self):
        """Test cross-validation with very small dataset."""
        arch_config = ModelArchitectureConfig(input_dim=5, n_components=3)
        config = CReSOConfiguration(architecture=arch_config)
        config.training.max_epochs = 2
        classifier = CReSOClassifier(config)
        
        # Very small dataset
        X, y = make_classification(n_samples=20, n_features=5, random_state=42)
        
        scores = cross_val_score(
            classifier, 
            X.astype(np.float32), 
            y.astype(np.int32), 
            cv=3
        )
        assert len(scores) == 3


if __name__ == "__main__":
    pytest.main([__file__])