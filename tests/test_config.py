"""Test suite for creso.config module."""

import pytest
import torch
from creso.config import (
    CReSOConfiguration,
    ModelArchitectureConfig,
    TrainingConfig,
    RegularizationConfig,
    FrequencySeedingConfig,
    SystemConfig,
)


class TestCReSOConfiguration:
    """Test CReSOConfiguration dataclass."""

    def test_default_initialization(self):
        """Test default config initialization."""
        arch_config = ModelArchitectureConfig(input_dim=10)
        config = CReSOConfiguration(architecture=arch_config)

        # Architecture parameters
        assert config.architecture.input_dim == 10
        assert config.architecture.n_components == 128
        assert config.architecture.use_localized_envelopes is True
        assert config.architecture.learn_envelope_centers is True
        assert config.architecture.initial_frequency_scale == 3.0
        assert config.architecture.geometric_hidden_dim == 64

        # Training hyperparameters
        assert config.training.learning_rate == 3e-3
        assert config.training.max_epochs == 25
        assert config.training.batch_size == 256
        assert config.training.weight_decay == 0.0
        assert config.training.frequency_dropout_probability == 0.2
        assert config.training.gradient_clip_norm == 1.0
        # AMP is disabled on CPU, so check the actual device-dependent behavior
        if config.system.device == "cpu":
            assert config.training.use_automatic_mixed_precision is False
        else:
            assert config.training.use_automatic_mixed_precision is True

        # Regularization
        assert config.regularization.l2_frequency_penalty == 1e-4
        assert config.regularization.group_l1_amplitude_penalty == 1e-3
        assert config.regularization.center_dispersion_penalty == 1e-5

        # Frequency seeding
        assert config.frequency_seeding.enable_frequency_seeding is True
        assert config.frequency_seeding.seeding_fraction == 0.2
        assert config.frequency_seeding.max_autocorr_lag == 12

        # System
        assert config.system.random_seed == 42
        assert config.system.num_workers == 4  # Check actual default
        assert config.system.pin_memory is True  # Check actual default

    def test_custom_initialization(self):
        """Test config with custom values."""
        arch_config = ModelArchitectureConfig(
            input_dim=5, n_components=64, use_localized_envelopes=False
        )
        train_config = TrainingConfig(learning_rate=1e-3, max_epochs=10)
        reg_config = RegularizationConfig(l2_frequency_penalty=1e-3)
        
        config = CReSOConfiguration(
            architecture=arch_config,
            training=train_config,
            regularization=reg_config,
        )

        assert config.architecture.input_dim == 5
        assert config.architecture.n_components == 64
        assert config.architecture.use_localized_envelopes is False
        assert config.training.learning_rate == 1e-3
        assert config.training.max_epochs == 10
        assert config.regularization.l2_frequency_penalty == 1e-3

    def test_validation_errors(self):
        """Test that invalid parameters raise errors."""
        from creso.exceptions import ValidationError
        
        with pytest.raises(ValidationError):
            ModelArchitectureConfig(input_dim=0)  # Must be positive
        
        with pytest.raises(ValidationError):
            ModelArchitectureConfig(input_dim=10, n_components=0)  # Must be positive
        
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0)  # Must be positive
        
        with pytest.raises(ValidationError):
            TrainingConfig(max_epochs=0)  # Must be positive

    def test_device_cuda_fallback(self):
        """Test device selection fallback."""
        if torch.cuda.is_available():
            sys_config = SystemConfig(device="cuda")
            assert sys_config.device == "cuda"
        else:
            # Should fallback to CPU when CUDA not available
            sys_config = SystemConfig(device="cuda")
            assert sys_config.device == "cpu"

    def test_to_dict_from_dict(self):
        """Test dictionary serialization."""
        arch_config = ModelArchitectureConfig(input_dim=10, n_components=64)
        config = CReSOConfiguration(architecture=arch_config)
        
        config_dict = config.to_dict()
        
        # Check structure
        assert "architecture" in config_dict
        assert "training" in config_dict
        assert "regularization" in config_dict
        assert "frequency_seeding" in config_dict
        assert "system" in config_dict
        
        # Check values
        assert config_dict["architecture"]["input_dim"] == 10
        assert config_dict["architecture"]["n_components"] == 64
        
        # Reconstruct from dict
        config_restored = CReSOConfiguration.from_dict(config_dict)
        assert config_restored.architecture.input_dim == 10
        assert config_restored.architecture.n_components == 64

    def test_preset_configurations(self):
        """Test preset configuration creation."""
        config_fast = CReSOConfiguration.from_preset("fast", input_dim=10)
        assert config_fast.architecture.input_dim == 10
        assert config_fast.name == "fast_preset"
        
        config_accurate = CReSOConfiguration.from_preset("accurate", input_dim=15)
        assert config_accurate.architecture.input_dim == 15
        assert config_accurate.name == "accurate_preset"
        
        # Test with overrides
        config_override = CReSOConfiguration.from_preset(
            "balanced", input_dim=20, learning_rate=1e-4
        )
        assert config_override.architecture.input_dim == 20
        assert config_override.training.learning_rate == 1e-4


class TestConfigComponents:
    """Test individual config components."""

    def test_architecture_config(self):
        """Test ModelArchitectureConfig."""
        config = ModelArchitectureConfig(
            input_dim=10,
            n_components=64,
            use_localized_envelopes=True,
            learn_envelope_centers=False,
            initial_frequency_scale=2.0,
            geometric_hidden_dim=32,
        )
        
        assert config.input_dim == 10
        assert config.n_components == 64
        assert config.use_localized_envelopes is True
        assert config.learn_envelope_centers is False
        assert config.initial_frequency_scale == 2.0
        assert config.geometric_hidden_dim == 32

    def test_training_config(self):
        """Test TrainingConfig."""
        config = TrainingConfig(
            learning_rate=1e-3,
            max_epochs=50,
            batch_size=128,
            weight_decay=1e-4,
            frequency_dropout_probability=0.1,
            gradient_clip_norm=0.5,
            use_automatic_mixed_precision=False,
            early_stopping_patience=10,
        )
        
        assert config.learning_rate == 1e-3
        assert config.max_epochs == 50
        assert config.batch_size == 128
        assert config.weight_decay == 1e-4
        assert config.frequency_dropout_probability == 0.1
        assert config.gradient_clip_norm == 0.5
        assert config.use_automatic_mixed_precision is False
        assert config.early_stopping_patience == 10

    def test_regularization_config(self):
        """Test RegularizationConfig."""
        config = RegularizationConfig(
            l2_frequency_penalty=1e-3,
            group_l1_amplitude_penalty=1e-2,
            center_dispersion_penalty=1e-6,
        )
        
        assert config.l2_frequency_penalty == 1e-3
        assert config.group_l1_amplitude_penalty == 1e-2
        assert config.center_dispersion_penalty == 1e-6

    def test_frequency_seeding_config(self):
        """Test FrequencySeedingConfig."""
        config = FrequencySeedingConfig(
            enable_frequency_seeding=False,
            seeding_fraction=0.5,
            max_autocorr_lag=24,
        )
        
        assert config.enable_frequency_seeding is False
        assert config.seeding_fraction == 0.5
        assert config.max_autocorr_lag == 24

    def test_system_config(self):
        """Test SystemConfig."""
        config = SystemConfig(
            device="cpu",
            random_seed=123,
            num_workers=4,
            pin_memory=True,
        )
        
        assert config.device == "cpu"
        assert config.random_seed == 123
        assert config.num_workers == 4
        assert config.pin_memory is True


if __name__ == "__main__":
    pytest.main([__file__])