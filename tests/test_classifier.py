"""
Tests for CReSO classifier functionality.
"""

import os
import tempfile
import numpy as np
import torch
import pytest
from unittest.mock import patch
from sklearn.datasets import make_classification, make_blobs
from sklearn.metrics import accuracy_score

from creso.config import ModelArchitectureConfig, CReSOConfiguration
from creso.classifier import CReSOClassifier, CReSOvRClassifier
from creso.exceptions import ValidationError


class TestCReSOClassifier:
    """Test CReSOClassifier functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        arch_config = ModelArchitectureConfig(input_dim=10, n_components=8)
        config = CReSOConfiguration(architecture=arch_config)
        config.training.max_epochs = 5
        config.training.batch_size = 32
        return config

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=2, 
            n_redundant=0, n_informative=8, random_state=42
        )
        return X.astype(np.float32), y.astype(np.int32)

    def test_initialization(self, config):
        """Test classifier initialization."""
        classifier = CReSOClassifier(config)
        assert classifier.config == config
        assert classifier.model is None
        assert classifier.standardizer is None
        assert not classifier._is_fitted

    def test_fit_basic(self, config, binary_data):
        """Test basic fitting functionality."""
        X, y = binary_data
        classifier = CReSOClassifier(config)
        
        # Fit the classifier
        result = classifier.fit(X, y)
        
        # Check return value and fitted state
        assert result is classifier  # Should return self
        assert classifier._is_fitted
        assert classifier.model is not None
        assert classifier.trainer is not None
        assert np.array_equal(classifier.classes_, [0, 1])
        assert classifier.n_features_in_ == 10

    def test_fit_with_validation(self, config, binary_data):
        """Test fitting with validation data."""
        X, y = binary_data
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]
        
        classifier = CReSOClassifier(config)
        classifier.fit(X_train, y_train, X_val, y_val)
        
        assert classifier._is_fitted
        assert classifier.n_features_in_ == 10

    def test_fit_no_standardization(self, config, binary_data):
        """Test fitting without standardization."""
        X, y = binary_data
        classifier = CReSOClassifier(config)
        classifier.fit(X, y, standardize=False)
        
        assert classifier._is_fitted
        assert classifier.standardizer is None

    def test_predict_basic(self, config, binary_data):
        """Test basic prediction functionality."""
        X, y = binary_data
        classifier = CReSOClassifier(config)
        classifier.fit(X, y)
        
        # Make predictions
        predictions = classifier.predict(X)
        
        # Check output format
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(X),)
        assert set(predictions).issubset({0, 1})
        
        # Should achieve reasonable accuracy
        accuracy = accuracy_score(y, predictions)
        assert accuracy > 0.6  # Should be better than random

    def test_predict_proba(self, config, binary_data):
        """Test probability prediction."""
        X, y = binary_data
        classifier = CReSOClassifier(config)
        classifier.fit(X, y)
        
        # Make probability predictions
        probas = classifier.predict_proba(X)
        
        # Check output format
        assert isinstance(probas, np.ndarray)
        assert probas.shape == (len(X), 2)  # Binary classification
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(probas >= 0) and np.all(probas <= 1)  # Valid probabilities

    def test_score(self, config, binary_data):
        """Test scoring functionality."""
        X, y = binary_data
        classifier = CReSOClassifier(config)
        classifier.fit(X, y)
        
        # Get score
        score = classifier.score(X, y)
        
        # Check score format and range
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.6  # Should achieve reasonable accuracy

    def test_decision_function(self, config, binary_data):
        """Test decision function."""
        X, y = binary_data
        classifier = CReSOClassifier(config)
        classifier.fit(X, y)
        
        # Get decision function values
        scores = classifier.decision_function(X)
        
        # Check output format
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (len(X),)
        
        # Check consistency with predictions
        predictions = classifier.predict(X)
        expected_predictions = (scores > 0).astype(int)
        assert np.array_equal(predictions, expected_predictions)

    def test_tensor_inputs(self, config, binary_data):
        """Test with PyTorch tensor inputs."""
        X, y = binary_data
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        
        classifier = CReSOClassifier(config)
        classifier.fit(X_tensor, y_tensor)
        
        predictions = classifier.predict(X_tensor)
        probas = classifier.predict_proba(X_tensor)
        
        assert isinstance(predictions, np.ndarray)
        assert isinstance(probas, np.ndarray)
        assert predictions.shape == (len(X),)
        assert probas.shape == (len(X), 2)

    def test_unfitted_errors(self, config):
        """Test errors when using unfitted classifier."""
        classifier = CReSOClassifier(config)
        X_dummy = np.random.randn(10, 10)
        
        with pytest.raises(ValidationError, match="must be fitted"):
            classifier.predict(X_dummy)
        
        with pytest.raises(ValidationError, match="must be fitted"):
            classifier.predict_proba(X_dummy)
        
        with pytest.raises(ValidationError, match="must be fitted"):
            classifier.score(X_dummy, np.ones(10))

    def test_input_validation(self, config, binary_data):
        """Test input validation during fitting."""
        X, y = binary_data
        classifier = CReSOClassifier(config)
        
        # Test mismatched shapes - sklearn checks first, so expect ValueError
        with pytest.raises(ValueError):
            classifier.fit(X, y[:-5])  # Fewer labels than samples
        
        # Test invalid labels
        y_invalid = np.array([0, 1, 2] * (len(y) // 3) + [0] * (len(y) % 3))
        with pytest.raises(ValidationError, match="binary classification"):
            classifier.fit(X, y_invalid)

    def test_save_load(self, config, binary_data):
        """Test save and load functionality."""
        X, y = binary_data
        classifier = CReSOClassifier(config)
        classifier.fit(X, y)
        
        # Get original predictions
        original_predictions = classifier.predict(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "classifier.pkl")
            
            # Save classifier
            classifier.save(save_path)
            assert os.path.exists(save_path)
            assert os.path.exists(save_path.replace(".pkl", "_model.pt"))
            
            # Load classifier
            loaded_classifier = CReSOClassifier.load(save_path)
            
            # Check loaded classifier
            assert loaded_classifier._is_fitted
            assert np.array_equal(loaded_classifier.classes_, [0, 1])
            assert loaded_classifier.n_features_in_ == 10
            
            # Check predictions match
            loaded_predictions = loaded_classifier.predict(X)
            assert np.array_equal(original_predictions, loaded_predictions)

    def test_torchscript_export(self, config, binary_data):
        """Test TorchScript export functionality."""
        X, y = binary_data
        classifier = CReSOClassifier(config)
        classifier.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "classifier.pt")
            
            # Test export
            classifier.to_torchscript(export_path, optimize=True, quantize=False)
            assert os.path.exists(export_path)
            
            # Test different options
            export_path_quantized = os.path.join(tmpdir, "classifier_quantized.pt")
            classifier.to_torchscript(
                export_path_quantized, 
                optimize=False, 
                quantize=True,
                return_probabilities=False
            )
            assert os.path.exists(export_path_quantized)

    @patch('creso.classifier.onnx')
    @patch('creso.classifier.ort')
    def test_onnx_export(self, mock_ort, mock_onnx, config, binary_data):
        """Test ONNX export functionality."""
        X, y = binary_data
        classifier = CReSOClassifier(config)
        classifier.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "classifier.onnx")
            
            # Test export (mocked to avoid dependency issues)
            try:
                classifier.to_onnx(
                    export_path, 
                    opset=17, 
                    optimize=True,
                    verify_model=False
                )
            except ImportError:
                # Expected if ONNX dependencies not available
                pass


class TestCReSOvRClassifier:
    """Test CReSOvRClassifier (One-vs-Rest) functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        arch_config = ModelArchitectureConfig(input_dim=8, n_components=6)
        config = CReSOConfiguration(architecture=arch_config)
        config.training.max_epochs = 3
        config.training.batch_size = 32
        return config

    @pytest.fixture
    def multiclass_data(self):
        """Generate multiclass classification data."""
        X, y = make_blobs(
            n_samples=150, centers=3, n_features=8, 
            random_state=42, cluster_std=2.0
        )
        return X.astype(np.float32), y.astype(np.int32)

    def test_initialization(self, config):
        """Test multiclass classifier initialization."""
        classifier = CReSOvRClassifier(config)
        assert classifier.config == config
        assert classifier.classifiers_ == {}
        assert classifier.classes_ is None

    def test_fit_multiclass(self, config, multiclass_data):
        """Test fitting multiclass classifier."""
        X, y = multiclass_data
        classifier = CReSOvRClassifier(config)
        
        # Fit the classifier
        result = classifier.fit(X, y)
        
        # Check return value and fitted state
        assert result is classifier
        assert len(classifier.classifiers_) == 3  # 3 classes
        assert len(classifier.classes_) == 3
        assert classifier.n_features_in_ == 8
        
        # Check all binary classifiers are fitted
        for binary_clf in classifier.classifiers_.values():
            assert binary_clf._is_fitted

    def test_predict_multiclass(self, config, multiclass_data):
        """Test multiclass prediction."""
        X, y = multiclass_data
        classifier = CReSOvRClassifier(config)
        classifier.fit(X, y)
        
        # Make predictions
        predictions = classifier.predict(X)
        
        # Check output format
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(X),)
        assert set(predictions).issubset(set(classifier.classes_))
        
        # Should achieve reasonable accuracy
        accuracy = accuracy_score(y, predictions)
        assert accuracy > 0.5  # Should be better than random for 3 classes

    def test_predict_proba_multiclass(self, config, multiclass_data):
        """Test multiclass probability prediction."""
        X, y = multiclass_data
        classifier = CReSOvRClassifier(config)
        classifier.fit(X, y)
        
        # Make probability predictions
        probas = classifier.predict_proba(X)
        
        # Check output format
        assert isinstance(probas, np.ndarray)
        assert probas.shape == (len(X), 3)  # 3 classes
        assert np.allclose(probas.sum(axis=1), 1.0, rtol=1e-5)  # Probabilities sum to 1
        assert np.all(probas >= 0) and np.all(probas <= 1)  # Valid probabilities

    def test_decision_function_multiclass(self, config, multiclass_data):
        """Test multiclass decision function."""
        X, y = multiclass_data
        classifier = CReSOvRClassifier(config)
        classifier.fit(X, y)
        
        # Get decision function values
        scores = classifier.decision_function(X)
        
        # Check output format
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (len(X), 3)  # One score per class
        
        # Check consistency with predictions
        predictions = classifier.predict(X)
        expected_predictions = classifier.classes_[np.argmax(scores, axis=1)]
        assert np.array_equal(predictions, expected_predictions)

    def test_save_load_multiclass(self, config, multiclass_data):
        """Test save and load for multiclass classifier."""
        X, y = multiclass_data
        classifier = CReSOvRClassifier(config)
        classifier.fit(X, y)
        
        # Get original predictions
        original_predictions = classifier.predict(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = tmpdir
            
            # Save classifier
            classifier.save(save_path)
            assert os.path.exists(os.path.join(save_path, "metadata.pkl"))
            
            # Load classifier
            loaded_classifier = CReSOvRClassifier.load(save_path)
            
            # Check loaded classifier
            assert len(loaded_classifier.classifiers_) == 3
            assert len(loaded_classifier.classes_) == 3
            assert loaded_classifier.n_features_in_ == 8
            
            # Check predictions match
            loaded_predictions = loaded_classifier.predict(X)
            assert np.array_equal(original_predictions, loaded_predictions)

    def test_unfitted_multiclass_errors(self, config):
        """Test errors when using unfitted multiclass classifier."""
        classifier = CReSOvRClassifier(config)
        X_dummy = np.random.randn(10, 8)
        
        with pytest.raises(ValidationError, match="must be fitted"):
            classifier.predict(X_dummy)
        
        with pytest.raises(ValidationError, match="must be fitted"):
            classifier.predict_proba(X_dummy)

    def test_empty_classifiers_error(self, config, multiclass_data):
        """Test error handling with no fitted classifiers."""
        X, y = multiclass_data
        classifier = CReSOvRClassifier(config)
        
        # Manually set to look fitted but with no classifiers
        classifier.classes_ = np.array([0, 1, 2])
        classifier.n_features_in_ = 8
        
        with pytest.raises(ValidationError, match="must be fitted"):
            classifier.predict(X)


if __name__ == "__main__":
    pytest.main([__file__])