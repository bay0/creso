"""
Tests for CReSO layers.
"""

import torch
import pytest
from creso.layers import WaveResonanceLayer
from creso.utils import set_global_seed


class TestWaveResonanceLayer:
    """Test WaveResonanceLayer functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        set_global_seed(42)
        self.input_dim = 5
        self.n_components = 16
        self.batch_size = 32
        self.layer = WaveResonanceLayer(
            input_dim=self.input_dim,
            n_components=self.n_components,
            localized=True,
            learn_centers=True,
        )

    def test_initialization(self):
        """Test layer initialization."""
        assert self.layer.input_dim == self.input_dim
        assert self.layer.n_components == self.n_components
        assert self.layer.localized is True
        assert self.layer.learn_centers is True

        # Check parameter shapes
        assert self.layer.omega.shape == (self.n_components, self.input_dim)
        assert self.layer.theta.shape == (self.n_components,)
        assert self.layer.a_c.shape == (self.n_components,)
        assert self.layer.a_s.shape == (self.n_components,)
        assert self.layer.centers.shape == (self.n_components, self.input_dim)
        assert self.layer.log_sigma.shape == (self.n_components,)

    def test_forward_shapes(self):
        """Test forward pass shapes."""
        x = torch.randn(self.batch_size, self.input_dim)

        z_spec, C, S, mask = self.layer(x, freq_dropout_p=0.0)

        assert z_spec.shape == (self.batch_size, 1)
        assert C.shape == (self.batch_size, self.n_components)
        assert S.shape == (self.batch_size, self.n_components)
        assert mask is None  # No dropout

    def test_forward_with_dropout(self):
        """Test forward pass with frequency dropout."""
        x = torch.randn(self.batch_size, self.input_dim)

        self.layer.train()
        z_spec, C, S, mask = self.layer(x, freq_dropout_p=0.5)

        assert z_spec.shape == (self.batch_size, 1)
        assert C.shape == (self.batch_size, self.n_components)
        assert S.shape == (self.batch_size, self.n_components)
        assert mask is not None
        assert mask.shape == (self.n_components,)
        assert mask.dtype == torch.bool

    def test_non_localized_layer(self):
        """Test non-localized layer."""
        layer = WaveResonanceLayer(
            input_dim=self.input_dim, n_components=self.n_components, localized=False
        )

        assert not hasattr(layer, "centers")
        assert not hasattr(layer, "log_sigma")

        x = torch.randn(self.batch_size, self.input_dim)
        z_spec, C, S, mask = layer(x)

        assert z_spec.shape == (self.batch_size, 1)

    def test_spectral_regularizers(self):
        """Test regularization terms."""
        reg_loss = self.layer.spectral_regularizers(
            l2_freq=1e-4, group_l1=1e-3, center_disp=1e-5
        )

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.numel() == 1
        assert reg_loss.item() >= 0
        assert torch.isfinite(reg_loss)

    def test_prune_by_amplitude(self):
        """Test amplitude-based pruning."""
        # Set some amplitudes to zero initially
        with torch.no_grad():
            self.layer.a_c[:5] = 0
            self.layer.a_s[:5] = 0

        torch.sum((self.layer.a_c != 0) | (self.layer.a_s != 0)).item()

        # Prune to top 8 components
        self.layer.prune_by_amplitude(top_k=8)

        final_nonzero = torch.sum((self.layer.a_c != 0) | (self.layer.a_s != 0)).item()

        assert final_nonzero <= 8

    def test_prune_by_threshold(self):
        """Test threshold-based pruning."""
        # Set known amplitudes
        with torch.no_grad():
            self.layer.a_c.fill_(0.1)
            self.layer.a_s.fill_(0.1)
            # Make some components larger
            self.layer.a_c[:5] = 0.5
            self.layer.a_s[:5] = 0.5

        # Prune with threshold
        threshold = 0.3
        self.layer.prune_by_amplitude(threshold=threshold)

        # Check that only large components remain
        amp_magnitudes = torch.sqrt(self.layer.a_c**2 + self.layer.a_s**2)
        nonzero_amps = amp_magnitudes[amp_magnitudes > 0]

        assert torch.all(nonzero_amps >= threshold)

    def test_frequency_magnitudes(self):
        """Test frequency magnitude computation."""
        freq_mags = self.layer.get_frequency_magnitudes()

        assert freq_mags.shape == (self.n_components,)
        assert torch.all(freq_mags >= 0)

    def test_amplitude_magnitudes(self):
        """Test amplitude magnitude computation."""
        amp_mags = self.layer.get_amplitude_magnitudes()

        assert amp_mags.shape == (self.n_components,)
        assert torch.all(amp_mags >= 0)

    def test_deterministic_forward(self):
        """Test that forward pass is deterministic with fixed seed."""
        x = torch.randn(self.batch_size, self.input_dim)

        # First forward pass
        set_global_seed(42)
        z1, _, _, _ = self.layer(x, freq_dropout_p=0.0)

        # Second forward pass with same seed
        set_global_seed(42)
        z2, _, _, _ = self.layer(x, freq_dropout_p=0.0)

        assert torch.allclose(z1, z2, atol=1e-6)

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        x = torch.randn(self.batch_size, self.input_dim, requires_grad=True)

        z_spec, _, _, _ = self.layer(x)
        loss = z_spec.sum()
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

        # Check that parameter gradients exist
        assert self.layer.omega.grad is not None
        assert self.layer.a_c.grad is not None
        assert self.layer.a_s.grad is not None


if __name__ == "__main__":
    pytest.main([__file__])
