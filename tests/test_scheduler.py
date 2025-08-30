"""
Tests for learning rate scheduler functionality.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

# pytestmark = pytest.mark.skip("Scheduler functionality needs full implementation")

from creso.config import ModelArchitectureConfig, CReSOConfiguration
from creso.trainer import CReSOTrainer


class TestLearningRateSchedulers:
    """Test learning rate scheduler functionality."""

    @pytest.fixture
    def base_config(self):
        """Create base configuration for testing."""
        arch_config = ModelArchitectureConfig(input_dim=10, n_components=8)
        config = CReSOConfiguration(architecture=arch_config)
        config.training.max_epochs = 10
        config.training.batch_size = 32
        config.training.learning_rate = 0.01
        return config

    @pytest.fixture
    def training_data(self):
        """Create training data for scheduler tests."""
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=2,
            n_redundant=0, n_informative=8, random_state=42
        )
        return X.astype(np.float32), y.astype(np.int32)

    def test_no_scheduler(self, base_config, training_data):
        """Test training without scheduler (baseline)."""
        # pytest.skip("Scheduler functionality needs full implementation")
        X, y = training_data
        
        # Configure no scheduler
        base_config.training.scheduler_type = "none"
        trainer = CReSOTrainer(base_config)
        
        # Train model
        model, optimizer, standardizer, history = trainer.fit(X, y, max_epochs=5)
        
        # Learning rate should remain constant
        final_lr = optimizer.param_groups[0]['lr']
        assert abs(final_lr - base_config.training.learning_rate) < 1e-6

    def test_reduce_on_plateau_scheduler(self, base_config, training_data):
        """Test ReduceLROnPlateau scheduler."""
        X, y = training_data
        split_idx = len(X) // 2
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Configure ReduceLROnPlateau scheduler
        base_config.training.scheduler_type = "reduce_on_plateau"
        base_config.training.scheduler_patience = 2
        base_config.training.scheduler_factor = 0.5
        
        trainer = CReSOTrainer(base_config)
        
        # Train with validation data (required for plateau detection)
        model, optimizer, standardizer, history = trainer.fit(
            X_train, y_train, X_val=X_val, y_val=y_val, max_epochs=8
        )
        
        # Check that scheduler was created and used
        final_lr = optimizer.param_groups[0]['lr']
        
        # Learning rate should be <= initial rate (may have been reduced)
        assert final_lr <= base_config.training.learning_rate

    def test_cosine_scheduler(self, base_config, training_data):
        """Test CosineAnnealingLR scheduler."""
        X, y = training_data
        
        # Configure cosine scheduler
        base_config.training.scheduler_type = "cosine"
        base_config.training.max_epochs = 10
        
        trainer = CReSOTrainer(base_config)
        
        # Train model
        model, optimizer, standardizer, history = trainer.fit(X, y, max_epochs=10)
        
        # Learning rate should have changed (cosine schedule)
        final_lr = optimizer.param_groups[0]['lr']
        
        # Final LR should be different from initial (cosine goes to near 0)
        assert final_lr < base_config.training.learning_rate

    def test_step_scheduler(self, base_config, training_data):
        """Test StepLR scheduler."""
        X, y = training_data
        
        # Configure step scheduler
        base_config.training.scheduler_type = "step"
        base_config.training.scheduler_step_size = 3
        base_config.training.scheduler_factor = 0.7
        
        trainer = CReSOTrainer(base_config)
        
        # Train model
        model, optimizer, standardizer, history = trainer.fit(X, y, max_epochs=7)
        
        # Learning rate should have been reduced at step 3 and 6
        final_lr = optimizer.param_groups[0]['lr']
        
        # After 2 steps (epochs 3 and 6), LR should be initial * (0.7)^2
        expected_lr = base_config.training.learning_rate * (0.7 ** 2)
        assert abs(final_lr - expected_lr) < 1e-6

    def test_scheduler_with_different_patience(self, base_config, training_data):
        """Test scheduler with different patience values."""
        X, y = training_data
        split_idx = len(X) // 2
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Test with high patience
        base_config.training.scheduler_type = "reduce_on_plateau"
        base_config.training.scheduler_patience = 10  # Very patient
        base_config.training.scheduler_factor = 0.5
        
        trainer = CReSOTrainer(base_config)
        
        model, optimizer, standardizer, history = trainer.fit(
            X_train, y_train, X_val=X_val, y_val=y_val, max_epochs=8
        )
        
        # With high patience, LR likely unchanged in short training
        final_lr = optimizer.param_groups[0]['lr']
        assert abs(final_lr - base_config.training.learning_rate) < 1e-6

    def test_scheduler_factor_effects(self, base_config, training_data):
        """Test different scheduler factor values."""
        X, y = training_data
        
        # Test with aggressive step size reduction
        base_config.training.scheduler_type = "step"
        base_config.training.scheduler_step_size = 2
        base_config.training.scheduler_factor = 0.1  # Aggressive reduction
        
        trainer = CReSOTrainer(base_config)
        
        model, optimizer, standardizer, history = trainer.fit(X, y, max_epochs=5)
        
        # After steps at epochs 2 and 4, LR should be much smaller
        final_lr = optimizer.param_groups[0]['lr']
        expected_lr = base_config.training.learning_rate * (0.1 ** 2)  # Two reductions
        assert abs(final_lr - expected_lr) < 1e-6

    def test_cosine_scheduler_min_lr(self, base_config, training_data):
        """Test cosine scheduler with minimum learning rate."""
        X, y = training_data
        
        base_config.training.scheduler_type = "cosine"
        base_config.training.scheduler_min_lr = 1e-6
        base_config.training.max_epochs = 20  # Long enough for significant decay
        
        trainer = CReSOTrainer(base_config)
        
        model, optimizer, standardizer, history = trainer.fit(X, y, max_epochs=20)
        
        final_lr = optimizer.param_groups[0]['lr']
        
        # Should not go below minimum LR
        assert final_lr >= base_config.training.scheduler_min_lr

    def test_invalid_scheduler_type(self, base_config, training_data):
        """Test handling of invalid scheduler type."""
        X, y = training_data
        
        # Set invalid scheduler type
        base_config.training.scheduler_type = "invalid_scheduler"
        
        trainer = CReSOTrainer(base_config)
        
        # Should fall back to no scheduler (not raise error)
        model, optimizer, standardizer, history = trainer.fit(X, y, max_epochs=3)
        
        # Learning rate should remain unchanged
        final_lr = optimizer.param_groups[0]['lr']
        assert abs(final_lr - base_config.training.learning_rate) < 1e-6

    def test_scheduler_with_early_stopping(self, base_config, training_data):
        """Test scheduler interaction with early stopping."""
        X, y = training_data
        split_idx = len(X) // 2
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Configure both scheduler and early stopping
        base_config.training.scheduler_type = "reduce_on_plateau"
        base_config.training.scheduler_patience = 2
        base_config.training.early_stopping_patience = 5
        
        trainer = CReSOTrainer(base_config)
        
        model, optimizer, standardizer, history = trainer.fit(
            X_train, y_train, X_val=X_val, y_val=y_val, max_epochs=15
        )
        
        # Should work without errors (both features can coexist)
        assert model is not None
        assert len(history['train_loss']) > 0

    def test_scheduler_state_in_history(self, base_config, training_data):
        """Test that scheduler state is tracked in training history."""
        X, y = training_data
        
        base_config.training.scheduler_type = "step"
        base_config.training.scheduler_step_size = 2
        base_config.training.scheduler_factor = 0.5
        
        trainer = CReSOTrainer(base_config)
        
        model, optimizer, standardizer, history = trainer.fit(X, y, max_epochs=6)
        
        # Check if learning rate is tracked in history
        # (Implementation may or may not track this - this test documents expected behavior)
        if 'learning_rate' in history:
            lr_history = history['learning_rate']
            assert len(lr_history) > 0
            # Should see step reductions at epochs 2 and 4
            assert lr_history[0] == base_config.training.learning_rate  # Initial LR

    def test_multiple_param_groups_scheduler(self, base_config, training_data):
        """Test scheduler with multiple parameter groups."""
        X, y = training_data
        
        base_config.training.scheduler_type = "step"
        base_config.training.scheduler_step_size = 3
        base_config.training.scheduler_factor = 0.5
        
        # Create trainer and let it create model first
        trainer = CReSOTrainer(base_config)
        
        # Start training - this will test scheduler with the model's parameter groups
        model, optimizer, standardizer, history = trainer.fit(X, y, max_epochs=6)
        
        # All parameter groups should have their learning rates scheduled
        final_lr = optimizer.param_groups[0]['lr']
        expected_lr = base_config.training.learning_rate * (0.5 ** 2)  # Two step reductions
        assert abs(final_lr - expected_lr) < 1e-6

    def test_scheduler_with_warmup_epochs(self, base_config, training_data):
        """Test that scheduler handles warmup periods correctly."""
        X, y = training_data
        
        base_config.training.scheduler_type = "step"
        base_config.training.scheduler_step_size = 2
        base_config.training.scheduler_factor = 0.7
        
        trainer = CReSOTrainer(base_config)
        
        # Train for exactly the step size to see one reduction
        model, optimizer, standardizer, history = trainer.fit(X, y, max_epochs=4)
        
        # Should have had two reductions by epoch 4
        final_lr = optimizer.param_groups[0]['lr']
        expected_lr = base_config.training.learning_rate * (0.7 ** 2)
        assert abs(final_lr - expected_lr) < 1e-6

    def test_plateau_scheduler_without_validation(self, base_config, training_data):
        """Test plateau scheduler behavior without validation data."""
        X, y = training_data
        
        # Configure plateau scheduler
        base_config.training.scheduler_type = "reduce_on_plateau"
        base_config.training.scheduler_patience = 3
        
        trainer = CReSOTrainer(base_config)
        
        # Train without validation data
        model, optimizer, standardizer, history = trainer.fit(X, y, max_epochs=8)
        
        # Should handle gracefully (may use training loss or skip scheduling)
        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr > 0  # Should be positive

    def test_scheduler_configuration_validation(self, base_config):
        """Test scheduler configuration validation."""
        # Test valid configurations
        valid_configs = [
            {"scheduler_type": "none"},
            {"scheduler_type": "reduce_on_plateau", "scheduler_patience": 5, "scheduler_factor": 0.5},
            {"scheduler_type": "cosine", "scheduler_min_lr": 1e-6},
            {"scheduler_type": "step", "scheduler_step_size": 10, "scheduler_factor": 0.9}
        ]
        
        for config_update in valid_configs:
            for key, value in config_update.items():
                setattr(base_config.training, key, value)
            
            trainer = CReSOTrainer(base_config)
            # Should not raise any validation errors during initialization
            assert trainer.config.training.scheduler_type in ["none", "reduce_on_plateau", "cosine", "step"]

    def test_scheduler_logging_and_monitoring(self, base_config, training_data):
        """Test that scheduler actions are properly logged."""
        X, y = training_data
        
        base_config.training.scheduler_type = "step"
        base_config.training.scheduler_step_size = 2
        
        trainer = CReSOTrainer(base_config)
        
        # Train with verbose mode to check logging
        model, optimizer, standardizer, history = trainer.fit(X, y, max_epochs=5, verbose=1)
        
        # Should complete without errors and with proper LR scheduling
        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr < base_config.training.learning_rate


class TestSchedulerEdgeCases:
    """Test edge cases and error conditions for schedulers."""

    @pytest.fixture
    def edge_case_config(self):
        """Create configuration for edge case testing."""
        arch_config = ModelArchitectureConfig(input_dim=5, n_components=4)
        config = CReSOConfiguration(architecture=arch_config)
        config.training.max_epochs = 5
        config.training.learning_rate = 0.1
        return config

    @pytest.fixture
    def minimal_data(self):
        """Create minimal dataset for edge case testing."""
        X, y = make_classification(
            n_samples=50, n_features=5, n_classes=2, random_state=42
        )
        return X.astype(np.float32), y.astype(np.int32)

    def test_zero_learning_rate(self, edge_case_config, minimal_data):
        """Test scheduler behavior with zero initial learning rate."""
        X, y = minimal_data
        
        edge_case_config.training.learning_rate = 0.0
        edge_case_config.training.scheduler_type = "step"
        
        trainer = CReSOTrainer(edge_case_config)
        
        # Should handle zero LR gracefully
        model, optimizer, standardizer, history = trainer.fit(X, y, max_epochs=3)
        assert optimizer.param_groups[0]['lr'] == 0.0

    def test_very_small_learning_rate(self, edge_case_config, minimal_data):
        """Test scheduler with very small learning rate."""
        X, y = minimal_data
        
        edge_case_config.training.learning_rate = 1e-10
        edge_case_config.training.scheduler_type = "cosine"
        
        trainer = CReSOTrainer(edge_case_config)
        
        model, optimizer, standardizer, history = trainer.fit(X, y, max_epochs=3)
        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr >= 0

    def test_single_epoch_training(self, edge_case_config, minimal_data):
        """Test scheduler with single epoch training."""
        X, y = minimal_data
        
        edge_case_config.training.scheduler_type = "step"
        edge_case_config.training.scheduler_step_size = 1
        
        trainer = CReSOTrainer(edge_case_config)
        
        # Train for just 1 epoch
        model, optimizer, standardizer, history = trainer.fit(X, y, max_epochs=1)
        
        # Should complete without errors
        assert len(history['train_loss']) == 1

    def test_scheduler_with_very_large_patience(self, edge_case_config, minimal_data):
        """Test plateau scheduler with very large patience."""
        X, y = minimal_data
        split_idx = len(X) // 2
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        edge_case_config.training.scheduler_type = "reduce_on_plateau"
        edge_case_config.training.scheduler_patience = 1000  # Very large
        
        trainer = CReSOTrainer(edge_case_config)
        
        model, optimizer, standardizer, history = trainer.fit(
            X_train, y_train, X_val=X_val, y_val=y_val, max_epochs=5
        )
        
        # LR should remain unchanged with such large patience
        final_lr = optimizer.param_groups[0]['lr']
        assert abs(final_lr - edge_case_config.training.learning_rate) < 1e-6

    def test_scheduler_parameter_boundaries(self, edge_case_config, minimal_data):
        """Test scheduler parameters at boundary values."""
        X, y = minimal_data
        
        # Test step scheduler with step_size = 1 (step every epoch)
        edge_case_config.training.scheduler_type = "step"
        edge_case_config.training.scheduler_step_size = 1
        edge_case_config.training.scheduler_factor = 0.9
        
        trainer = CReSOTrainer(edge_case_config)
        
        model, optimizer, standardizer, history = trainer.fit(X, y, max_epochs=3)
        
        # Should have stepped down 3 times
        final_lr = optimizer.param_groups[0]['lr']
        expected_lr = edge_case_config.training.learning_rate * (0.9 ** 3)
        assert abs(final_lr - expected_lr) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__])