"""
Tests for CReSO model.
"""

import torch
import pytest
import tempfile
import os
from creso.config import CReSOConfiguration, ModelArchitectureConfig, TrainingConfig
from creso.model import CReSOModel
from creso.utils import set_global_seed, Standardizer


class TestCReSOModel:
    """Test CReSOModel functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        set_global_seed(42)
        arch_config = ModelArchitectureConfig(input_dim=10, n_components=32)
        train_config = TrainingConfig(max_epochs=5)
        self.config = CReSOConfiguration(architecture=arch_config, training=train_config)
        self.model = CReSOModel(self.config)
        self.batch_size = 16

    def test_initialization(self):
        """Test model initialization."""
        assert self.model.input_dim == self.config.architecture.input_dim
        assert hasattr(self.model, "wave_layer")
        assert hasattr(self.model, "geom_net")
        assert hasattr(self.model, "gate")

        # Check wave layer config
        assert self.model.wave_layer.input_dim == self.config.architecture.input_dim
        assert self.model.wave_layer.n_components == self.config.architecture.n_components

    def test_forward_shapes(self):
        """Test forward pass shapes."""
        x = torch.randn(self.batch_size, self.config.architecture.input_dim)

        z, z_spec, z_geom, alpha, (C, S, mask) = self.model(x, train_mode=False)

        assert z.shape == (self.batch_size, 1)
        assert z_spec.shape == (self.batch_size, 1)
        assert z_geom.shape == (self.batch_size, 1)
        assert alpha.shape == (self.batch_size, 1)
        assert C.shape == (self.batch_size, self.config.architecture.n_components)
        assert S.shape == (self.batch_size, self.config.architecture.n_components)
        assert mask is None  # No dropout in eval mode

    def test_forward_train_mode(self):
        """Test forward pass in training mode."""
        x = torch.randn(self.batch_size, self.config.architecture.input_dim)

        self.model.train()
        z, z_spec, z_geom, alpha, (C, S, mask) = self.model(x, train_mode=True)

        assert z.shape == (self.batch_size, 1)
        assert mask is not None  # Dropout applied in train mode

    def test_gating_mechanism(self):
        """Test gating mechanism behavior."""
        x = torch.randn(self.batch_size, self.config.architecture.input_dim)

        z, z_spec, z_geom, alpha, _ = self.model(x, train_mode=False)

        # Alpha should be between 0 and 1 (sigmoid output)
        assert torch.all(alpha >= 0)
        assert torch.all(alpha <= 1)

        # Manual combination check
        z_manual = alpha * z_spec + (1 - alpha) * z_geom
        assert torch.allclose(z, z_manual, atol=1e-6)

    def test_regularization(self):
        """Test regularization computation."""
        reg_loss = self.model.regularization()

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.numel() == 1
        assert reg_loss.item() >= 0
        assert torch.isfinite(reg_loss)

        # Test with custom parameters
        reg_loss_custom = self.model.regularization(
            l2_freq=1e-3, group_l1=1e-2, center_disp=1e-4
        )
        assert isinstance(reg_loss_custom, torch.Tensor)

    def test_pruning(self):
        """Test spectral component pruning."""
        # Get initial amplitude magnitudes
        initial_amps = self.model.wave_layer.get_amplitude_magnitudes()
        torch.sum(initial_amps > 0).item()

        # Prune to half the components
        target_k = self.config.architecture.n_components // 2
        self.model.prune_spectral_components(top_k=target_k)

        # Check pruning effect
        final_amps = self.model.wave_layer.get_amplitude_magnitudes()
        final_nonzero = torch.sum(final_amps > 0).item()

        assert final_nonzero <= target_k

    def test_spectral_info(self):
        """Test spectral information extraction."""
        info = self.model.get_spectral_info()

        required_keys = [
            "freq_magnitudes",
            "amp_magnitudes",
            "frequencies",
            "phases",
            "cos_amps",
            "sin_amps",
        ]

        for key in required_keys:
            assert key in info
            assert isinstance(info[key], torch.Tensor)

        # Check shapes
        assert info["freq_magnitudes"].shape == (self.config.architecture.n_components,)
        assert info["amp_magnitudes"].shape == (self.config.architecture.n_components,)
        assert info["frequencies"].shape == (self.config.architecture.n_components, self.config.architecture.input_dim)
        assert info["phases"].shape == (self.config.architecture.n_components,)
        assert info["cos_amps"].shape == (self.config.architecture.n_components,)
        assert info["sin_amps"].shape == (self.config.architecture.n_components,)

    def test_deterministic_forward(self):
        """Test deterministic forward pass."""
        x = torch.randn(self.batch_size, self.config.architecture.input_dim)

        # First forward pass
        set_global_seed(42)
        z1, _, _, _, _ = self.model(x, train_mode=False)

        # Second forward pass with same seed
        set_global_seed(42)
        z2, _, _, _, _ = self.model(x, train_mode=False)

        assert torch.allclose(z1, z2, atol=1e-6)

    def test_save_load(self):
        """Test model save and load functionality."""
        # Create test data
        x = torch.randn(self.batch_size, self.config.architecture.input_dim)

        # Get initial prediction
        z1, _, _, _, _ = self.model(x, train_mode=False)

        # Create standardizer
        standardizer = Standardizer()
        standardizer.fit(x)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model.pt")
            extra_data = {"test_key": "test_value"}

            self.model.save(save_path, standardizer, extra_data)
            assert os.path.exists(save_path)

            # Load model
            loaded_model, loaded_standardizer, loaded_extra = CReSOModel.load(save_path)

            # Check loaded model gives same prediction
            loaded_model.eval()
            z2, _, _, _, _ = loaded_model(x, train_mode=False)
            assert torch.allclose(z1, z2, atol=1e-6)

            # Check loaded standardizer
            assert loaded_standardizer is not None
            assert loaded_standardizer.fitted

            # Check extra data
            assert loaded_extra["test_key"] == "test_value"

    def test_torchscript_export(self):
        """Test TorchScript export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "model.ts")

            # Export to TorchScript
            self.model.to_torchscript(export_path)
            assert os.path.exists(export_path)

            # Load and test
            loaded_model = torch.jit.load(export_path)

            x = torch.randn(1, self.config.architecture.input_dim)
            original_out, _, _, _, _ = self.model(x, train_mode=False)
            traced_out = loaded_model(x)

            assert torch.allclose(original_out, traced_out, atol=1e-5)

    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        x = torch.randn(self.batch_size, self.config.architecture.input_dim, requires_grad=True)

        z, _, _, _, _ = self.model(x, train_mode=True)
        loss = z.sum() + self.model.regularization()
        loss.backward()

        # Check input gradients
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

        # Check parameter gradients exist
        for param in self.model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_different_input_sizes(self):
        """Test model with different batch sizes."""
        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, self.config.architecture.input_dim)
            z, _, _, _, _ = self.model(x, train_mode=False)
            assert z.shape == (batch_size, 1)


if __name__ == "__main__":
    pytest.main([__file__])
