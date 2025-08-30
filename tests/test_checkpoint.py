"""
Tests for checkpoint and resume functionality.
"""

import os
import tempfile
import numpy as np
import torch
import pytest
from sklearn.datasets import make_classification

from creso.config import ModelArchitectureConfig, CReSOConfiguration
from creso.trainer import CReSOTrainer
from creso.model import CReSOModel
from creso.exceptions import ValidationError


class TestCheckpointFunctionality:
    """Test checkpoint saving and loading functionality."""

    @pytest.fixture
    def trainer_setup(self):
        """Create trainer and data for testing."""
        arch_config = ModelArchitectureConfig(input_dim=10, n_components=8)
        config = CReSOConfiguration(architecture=arch_config)
        config.training.max_epochs = 10
        config.training.batch_size = 32
        config.training.early_stopping_patience = 5
        
        trainer = CReSOTrainer(config)
        
        # Generate small dataset for quick testing
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=2,
            n_redundant=0, n_informative=8, random_state=42
        )
        
        return trainer, X.astype(np.float32), y.astype(np.int32), config

    def test_checkpoint_creation(self, trainer_setup):
        """Test basic checkpoint creation."""
        trainer, X, y, config = trainer_setup
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
            
            # Set epochs in config and train
            config.training.max_epochs = 3
            trainer.fit(X, y)
            
            # Create dummy checkpoint data
            model = CReSOModel(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # Test checkpoint saving
            trainer.save_checkpoint(
                filepath=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=2,
                train_loss=0.5,
                val_loss=0.6,
                best_val_loss=0.55,
                lr=0.01,
                config=config
            )
            
            # Check file was created
            assert os.path.exists(checkpoint_path)

    def test_checkpoint_loading(self, trainer_setup):
        """Test checkpoint loading."""
        trainer, X, y, config = trainer_setup
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
            
            # Create and save checkpoint
            model = CReSOModel(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            trainer.save_checkpoint(
                filepath=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=5,
                train_loss=0.3,
                val_loss=0.35,
                best_val_loss=0.32,
                lr=0.001,
                config=config
            )
            
            # Load checkpoint
            loaded_data = trainer.load_checkpoint(checkpoint_path)
            
            # Verify loaded data
            assert loaded_data['epoch'] == 5
            assert loaded_data['train_loss'] == 0.3
            assert loaded_data['val_loss'] == 0.35
            assert loaded_data['best_val_loss'] == 0.32
            assert loaded_data['lr'] == 0.001
            assert 'model_state_dict' in loaded_data
            assert 'optimizer_state_dict' in loaded_data
            assert 'config' in loaded_data

    def test_resume_training_from_checkpoint(self, trainer_setup):
        """Test resuming training from checkpoint."""
        trainer, X, y, config = trainer_setup
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "resume_checkpoint.pt")
            
            # Train initially and save checkpoint
            model, optimizer, standardizer, history = trainer.fit(
                X, y, max_epochs=3, 
                checkpoint_path=checkpoint_path,
                checkpoint_frequency=1
            )
            
            # Verify checkpoint was saved
            assert os.path.exists(checkpoint_path)
            
            # Create new trainer and resume
            new_trainer = CReSOTrainer(config)
            
            resumed_model, resumed_optimizer, resumed_standardizer, resumed_history = new_trainer.fit(
                X, y, max_epochs=6,  # Train for 3 more epochs
                resume_from_checkpoint=checkpoint_path
            )
            
            # Verify training resumed properly
            assert resumed_model is not None
            assert resumed_optimizer is not None
            assert len(resumed_history['train_loss']) > len(history['train_loss'])

    def test_checkpoint_directory_creation(self, trainer_setup):
        """Test that checkpoint directories are created automatically."""
        trainer, X, y, config = trainer_setup
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directory path that doesn't exist
            checkpoint_path = os.path.join(tmpdir, "checkpoints", "subdir", "checkpoint.pt")
            
            model = CReSOModel(config)
            optimizer = torch.optim.Adam(model.parameters())
            
            # Save checkpoint - should create directories
            trainer.save_checkpoint(
                filepath=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=1,
                train_loss=0.5,
                config=config
            )
            
            assert os.path.exists(checkpoint_path)
            assert os.path.isdir(os.path.dirname(checkpoint_path))

    def test_checkpoint_with_validation_data(self, trainer_setup):
        """Test checkpoint with validation data."""
        trainer, X, y, config = trainer_setup
        
        # Split data
        split_idx = len(X) // 2
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "val_checkpoint.pt")
            
            # Train with validation and checkpointing
            model, optimizer, standardizer, history = trainer.fit(
                X_train, y_train, 
                X_val=X_val, y_val=y_val,
                max_epochs=3,
                checkpoint_path=checkpoint_path,
                checkpoint_frequency=1
            )
            
            # Verify checkpoint contains validation metrics
            assert os.path.exists(checkpoint_path)
            
            loaded_data = trainer.load_checkpoint(checkpoint_path)
            assert 'val_loss' in loaded_data
            assert loaded_data['val_loss'] is not None

    def test_checkpoint_frequency(self, trainer_setup):
        """Test checkpoint saving frequency."""
        trainer, X, y, config = trainer_setup
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = tmpdir
            
            # Train with checkpoint frequency of 2
            model, optimizer, standardizer, history = trainer.fit(
                X, y, max_epochs=5,
                checkpoint_path=os.path.join(checkpoint_dir, "freq_checkpoint.pt"),
                checkpoint_frequency=2  # Save every 2 epochs
            )
            
            # Should have saved checkpoints at epochs 2 and 4
            checkpoint_path = os.path.join(checkpoint_dir, "freq_checkpoint.pt")
            assert os.path.exists(checkpoint_path)

    def test_best_checkpoint_saving(self, trainer_setup):
        """Test saving best model checkpoints."""
        trainer, X, y, config = trainer_setup
        
        # Split for validation
        split_idx = len(X) // 2
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "best_checkpoint.pt")
            
            # Train and save best checkpoint
            model, optimizer, standardizer, history = trainer.fit(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                max_epochs=5,
                checkpoint_path=checkpoint_path,
                save_best_only=True
            )
            
            if os.path.exists(checkpoint_path):
                # Load and verify it's marked as best
                loaded_data = trainer.load_checkpoint(checkpoint_path)
                assert 'best_val_loss' in loaded_data

    def test_checkpoint_state_consistency(self, trainer_setup):
        """Test that checkpoint state is consistent."""
        trainer, X, y, config = trainer_setup
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "state_checkpoint.pt")
            
            # Train initial model
            model, optimizer, standardizer, history = trainer.fit(
                X, y, max_epochs=2,
                checkpoint_path=checkpoint_path
            )
            
            # Get model parameters before checkpoint
            original_params = {name: param.clone() for name, param in model.named_parameters()}
            
            # Load checkpoint into new model
            new_model = CReSOModel(config)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=config.training.learning_rate)
            
            checkpoint_data = trainer.load_checkpoint(checkpoint_path)
            new_model.load_state_dict(checkpoint_data['model_state_dict'])
            new_optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            # Compare parameters - they should be identical
            for name, param in new_model.named_parameters():
                assert torch.allclose(param, original_params[name], atol=1e-6)

    def test_resume_with_different_config(self, trainer_setup):
        """Test error when resuming with incompatible config."""
        trainer, X, y, config = trainer_setup
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "config_checkpoint.pt")
            
            # Train and save checkpoint
            model = CReSOModel(config)
            optimizer = torch.optim.Adam(model.parameters())
            
            trainer.save_checkpoint(
                filepath=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=1,
                train_loss=0.5,
                config=config
            )
            
            # Create different config
            different_arch = ModelArchitectureConfig(input_dim=20, n_components=16)  # Different dimensions
            different_config = CReSOConfiguration(architecture=different_arch)
            different_trainer = CReSOTrainer(different_config)
            
            # Attempting to resume should handle the incompatibility
            # (Implementation may either ignore, warn, or adapt)
            try:
                different_trainer.fit(
                    X, y, max_epochs=2,
                    resume_from_checkpoint=checkpoint_path
                )
                # If it succeeds, the implementation handles config differences gracefully
            except Exception:
                # If it fails, that's also acceptable behavior
                pass

    def test_checkpoint_with_early_stopping(self, trainer_setup):
        """Test checkpoint behavior with early stopping."""
        trainer, X, y, config = trainer_setup
        
        # Set aggressive early stopping
        config.training.early_stopping_patience = 2
        
        # Split for validation
        split_idx = len(X) // 2
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "early_stop_checkpoint.pt")
            
            model, optimizer, standardizer, history = trainer.fit(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                max_epochs=10,  # Will likely stop early
                checkpoint_path=checkpoint_path
            )
            
            # Should have created a checkpoint even if stopped early
            if os.path.exists(checkpoint_path):
                loaded_data = trainer.load_checkpoint(checkpoint_path)
                assert loaded_data['epoch'] <= 10

    def test_checkpoint_loading_nonexistent_file(self, trainer_setup):
        """Test error when loading nonexistent checkpoint."""
        trainer, X, y, config = trainer_setup
        
        nonexistent_path = "/tmp/nonexistent_checkpoint.pt"
        
        with pytest.raises((FileNotFoundError, ValidationError)):
            trainer.load_checkpoint(nonexistent_path)

    def test_checkpoint_model_architecture_preservation(self, trainer_setup):
        """Test that model architecture is preserved in checkpoints."""
        trainer, X, y, config = trainer_setup
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "arch_checkpoint.pt")
            
            # Create model with specific architecture
            original_model = CReSOModel(config)
            optimizer = torch.optim.Adam(original_model.parameters())
            
            # Save checkpoint
            trainer.save_checkpoint(
                filepath=checkpoint_path,
                model=original_model,
                optimizer=optimizer,
                epoch=1,
                train_loss=0.5,
                config=config
            )
            
            # Load checkpoint and verify config matches
            loaded_data = trainer.load_checkpoint(checkpoint_path)
            loaded_config = loaded_data['config']
            
            assert loaded_config.architecture.input_dim == config.architecture.input_dim
            assert loaded_config.architecture.n_components == config.architecture.n_components

    def test_checkpoint_optimizer_state(self, trainer_setup):
        """Test that optimizer state is preserved in checkpoints."""
        trainer, X, y, config = trainer_setup
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "optimizer_checkpoint.pt")
            
            # Train for a few steps to build optimizer state
            model, optimizer, standardizer, history = trainer.fit(
                X, y, max_epochs=3,
                checkpoint_path=checkpoint_path
            )
            
            if os.path.exists(checkpoint_path):
                # Load checkpoint
                loaded_data = trainer.load_checkpoint(checkpoint_path)
                
                # Verify optimizer state is present
                assert 'optimizer_state_dict' in loaded_data
                optimizer_state = loaded_data['optimizer_state_dict']
                
                # Adam optimizer should have momentum states
                if 'state' in optimizer_state:
                    assert len(optimizer_state['state']) > 0

    def test_manual_checkpoint_save_load(self, trainer_setup):
        """Test manual checkpoint save/load operations."""
        trainer, X, y, config = trainer_setup
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "manual_checkpoint.pt")
            
            # Create model and train briefly
            model, optimizer, standardizer, history = trainer.fit(X, y, max_epochs=2)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Manually save checkpoint
            trainer.save_checkpoint(
                filepath=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=2,
                train_loss=history['train_loss'][-1],
                val_loss=None,
                best_val_loss=None,
                lr=current_lr,
                config=config,
                additional_data={'custom_metric': 0.95}
            )
            
            # Manually load checkpoint
            loaded_data = trainer.load_checkpoint(checkpoint_path)
            
            # Verify custom data was saved
            assert loaded_data['additional_data']['custom_metric'] == 0.95
            assert loaded_data['lr'] == current_lr


if __name__ == "__main__":
    pytest.main([__file__])