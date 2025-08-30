"""
Tests for CReSO trainer.
"""

import torch
import pytest
import numpy as np
from sklearn.datasets import make_classification

from creso.config import CReSOConfiguration, ModelArchitectureConfig, TrainingConfig
from creso.trainer import CReSOTrainer, acf_peak_seeds, apply_freq_seeds
from creso.model import CReSOModel
from creso.utils import set_global_seed


class TestACFSeeding:
    """Test ACF-based frequency seeding."""

    def test_acf_peak_seeds(self):
        """Test ACF peak extraction."""
        set_global_seed(42)

        # Create synthetic periodic data
        t = np.linspace(0, 10, 100)
        input_dim = 3
        X = np.zeros((100, input_dim))

        # Add periodic components
        X[:, 0] = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz
        X[:, 1] = np.cos(2 * np.pi * 2.0 * t)  # 2 Hz
        X[:, 2] = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz

        seeds = acf_peak_seeds(X, input_dim, n_modes=6, max_lag=20)

        assert isinstance(seeds, list)
        assert len(seeds) <= 6

        # Check seed format
        for dim, freq in seeds:
            assert isinstance(dim, int)
            assert isinstance(freq, float)
            assert 0 <= dim < input_dim
            assert freq > 0

    def test_apply_freq_seeds(self):
        """Test frequency seeding application."""
        set_global_seed(42)

        arch_config = ModelArchitectureConfig(input_dim=5, n_components=16)
        config = CReSOConfiguration(architecture=arch_config)
        model = CReSOModel(config)

        # Create synthetic data
        X_train = torch.randn(100, 5)

        # Get initial frequencies
        initial_freqs = model.wave_layer.omega.clone()

        # Apply seeding
        apply_freq_seeds(model, X_train, frac=0.25)

        # Check that some frequencies were modified
        final_freqs = model.wave_layer.omega

        # At least some frequencies should have changed
        # (though not guaranteed due to randomness in ACF)
        assert not torch.allclose(initial_freqs, final_freqs, atol=1e-6)


class TestCReSOTrainer:
    """Test CReSOTrainer functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        set_global_seed(42)
        arch_config = ModelArchitectureConfig(input_dim=10, n_components=32)
        train_config = TrainingConfig(max_epochs=3, learning_rate=1e-2, batch_size=32)  # Quick training for tests
        self.config = CReSOConfiguration(architecture=arch_config, training=train_config)
        self.trainer = CReSOTrainer(self.config)

        # Create synthetic classification data
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=1,
            random_state=42,
        )

        self.X_train = X[:150]
        self.y_train = y[:150]
        self.X_test = X[150:]
        self.y_test = y[150:]

    def test_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.config == self.config

    def test_fit_basic(self):
        """Test basic model fitting."""
        model, optimizer, standardizer, history = self.trainer.fit(
            self.X_train, self.y_train, standardize=True
        )

        assert isinstance(model, CReSOModel)
        assert standardizer is not None
        assert standardizer.fitted

        # Model should be in eval mode after training
        assert not model.training

    def test_fit_with_validation(self):
        """Test fitting with validation data."""
        model, optimizer, standardizer, history = self.trainer.fit(
            self.X_train,
            self.y_train,
            X_val=self.X_test,
            y_val=self.y_test,
            standardize=True,
        )

        assert isinstance(model, CReSOModel)
        assert standardizer is not None

    def test_fit_no_standardization(self):
        """Test fitting without standardization."""
        model, optimizer, standardizer, history = self.trainer.fit(
            self.X_train, self.y_train, standardize=False
        )

        assert isinstance(model, CReSOModel)
        assert standardizer is None

    def test_class_weighting(self):
        """Test class weighting options."""
        # Balanced weighting
        model1, _, _, _ = self.trainer.fit(
            self.X_train, self.y_train, class_weight="balanced"
        )

        # Custom weight
        model2, _, _, _ = self.trainer.fit(self.X_train, self.y_train, class_weight=2.0)

        # Dict weighting
        model3, _, _, _ = self.trainer.fit(
            self.X_train, self.y_train, class_weight={0: 1.0, 1: 2.0}
        )

        assert isinstance(model1, CReSOModel)
        assert isinstance(model2, CReSOModel)
        assert isinstance(model3, CReSOModel)

    def test_prediction_methods(self):
        """Test prediction methods."""
        model, optimizer, standardizer, history = self.trainer.fit(
            self.X_train, self.y_train, standardize=True
        )

        # Test predict_proba
        proba = self.trainer.predict_proba(model, self.X_test, standardizer)
        assert proba.shape == (len(self.X_test), 2)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1.0)

        # Test predict
        preds = self.trainer.predict(model, self.X_test, standardizer)
        assert preds.shape == (len(self.X_test),)
        assert np.all(np.isin(preds, [0, 1]))

        # Test with custom threshold
        preds_custom = self.trainer.predict(
            model, self.X_test, standardizer, threshold=0.7
        )
        assert preds_custom.shape == (len(self.X_test),)

    def test_training_achieves_reasonable_accuracy(self):
        """Test that training achieves reasonable accuracy on simple data."""
        # Create easier classification task
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=2,
            class_sep=2.0,  # Well-separated classes
            random_state=42,
        )

        X_train, _X_test = X[:200], X[200:]
        y_train, _y_test = y[:200], y[200:]

        # Train with more epochs for better convergence
        arch_config = ModelArchitectureConfig(input_dim=10, n_components=64)
        train_config = TrainingConfig(max_epochs=10, learning_rate=1e-2)
        config = CReSOConfiguration(architecture=arch_config, training=train_config)
        trainer = CReSOTrainer(config)

        model, optimizer, standardizer, history = trainer.fit(X_train, y_train)

        # Check training accuracy
        train_preds = trainer.predict(model, X_train, standardizer)
        train_acc = np.mean(train_preds == y_train)

        # Should achieve decent accuracy on this simple task
        assert train_acc > 0.7, f"Training accuracy too low: {train_acc:.3f}"

    def test_tensor_inputs(self):
        """Test training with tensor inputs."""
        X_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_train, dtype=torch.float32)

        model, optimizer, standardizer, history = self.trainer.fit(X_tensor, y_tensor)

        assert isinstance(model, CReSOModel)

    def test_early_stopping(self):
        """Test early stopping with validation."""
        # Create config with many epochs
        arch_config = ModelArchitectureConfig(input_dim=10, n_components=32)
        train_config = TrainingConfig(max_epochs=50)
        config = CReSOConfiguration(architecture=arch_config, training=train_config)
        trainer = CReSOTrainer(config)

        model, optimizer, standardizer, history = trainer.fit(
            self.X_train, self.y_train, X_val=self.X_test, y_val=self.y_test
        )

        # Training should complete (early stopping or full epochs)
        assert isinstance(model, CReSOModel)

    def test_different_data_shapes(self):
        """Test trainer with different data shapes."""
        # 1D labels (should work)
        y_1d = self.y_train.reshape(-1)
        model1, _, _, _ = self.trainer.fit(self.X_train, y_1d)

        # 2D labels (should be squeezed)
        y_2d = self.y_train.reshape(-1, 1)
        model2, _, _, _ = self.trainer.fit(self.X_train, y_2d)

        assert isinstance(model1, CReSOModel)
        assert isinstance(model2, CReSOModel)


if __name__ == "__main__":
    pytest.main([__file__])
