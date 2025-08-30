"""
Tests for CReSO adapters.
"""

import pytest
import numpy as np
import torch
from creso.adapters.timeseries import make_multirate_windows, TimeSeriesCReSOClassifier
from creso.config import CReSOConfiguration, ModelArchitectureConfig, TrainingConfig
from creso.utils import set_global_seed

# Graph tests only run if scipy is available
try:
    from creso.adapters.graph import (
        normalize_laplacian,
        chebyshev_polynomials,
        graph_wave_features,
        GraphNodeCReSOClassifier,
    )
    import scipy.sparse as sp

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class TestTimeSeriesAdapter:
    """Test time-series adapter functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        set_global_seed(42)

        # Create synthetic time series
        self.T = 200
        t = np.linspace(0, 10, self.T)

        # Multi-component signal
        self.series = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.cos(2 * np.pi * 3.0 * t)

        # Binary classification target
        self.target = (self.series > 0).astype(int)

        # Multivariate series
        self.mv_series = np.column_stack([self.series, np.cos(2 * np.pi * 0.5 * t)])

    def test_make_multirate_windows_univariate(self):
        """Test multi-rate windowing for univariate series."""
        X, y = make_multirate_windows(
            self.series, self.target, window=32, horizon=1, rates=[1, 2, 4], step=1
        )

        assert X is not None
        assert y is not None
        assert X.ndim == 2
        assert y.ndim == 1
        assert len(X) == len(y)

        # Feature dimension should be window * num_rates * series_dim
        expected_features = 32 * 3 * 1  # window * rates * 1D
        assert X.shape[1] == expected_features

    def test_make_multirate_windows_multivariate(self):
        """Test multi-rate windowing for multivariate series."""
        X, y = make_multirate_windows(
            self.mv_series, self.target, window=32, horizon=1, rates=[1, 2], step=2
        )

        assert X is not None
        assert y is not None

        # Feature dimension should be window * num_rates * series_dim
        expected_features = 32 * 2 * 2  # window * rates * 2D
        assert X.shape[1] == expected_features

    def test_make_multirate_windows_no_target(self):
        """Test windowing without targets."""
        X, y = make_multirate_windows(
            self.series, None, window=32, horizon=1, rates=[1, 2], step=4
        )

        assert X is not None
        assert y is None
        assert X.ndim == 2

    def test_make_multirate_windows_tensor_input(self):
        """Test windowing with tensor inputs."""
        series_tensor = torch.tensor(self.series, dtype=torch.float32)
        target_tensor = torch.tensor(self.target, dtype=torch.float32)

        X, y = make_multirate_windows(
            series_tensor, target_tensor, window=16, horizon=1, rates=[1, 2]
        )

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_make_multirate_windows_insufficient_data(self):
        """Test windowing with insufficient data."""
        short_series = self.series[:10]
        short_target = self.target[:10]

        with pytest.raises(ValueError, match="Series too short"):
            make_multirate_windows(
                short_series, short_target, window=32, horizon=1, rates=[1, 2, 4]
            )

    def test_timeseries_classifier_initialization(self):
        """Test time-series classifier initialization."""
        arch_config = ModelArchitectureConfig(input_dim=1)
        train_config = TrainingConfig(max_epochs=3)
        config = CReSOConfiguration(architecture=arch_config, training=train_config)

        clf = TimeSeriesCReSOClassifier(
            window=32, horizon=1, rates=[1, 2], config=config
        )

        assert clf.window == 32
        assert clf.horizon == 1
        assert clf.rates == [1, 2]
        assert clf.classifier is None

    def test_timeseries_classifier_fit_predict(self):
        """Test time-series classifier fit and predict."""
        arch_config = ModelArchitectureConfig(input_dim=1, n_components=16)
        train_config = TrainingConfig(max_epochs=3)
        config = CReSOConfiguration(architecture=arch_config, training=train_config)

        clf = TimeSeriesCReSOClassifier(
            window=32, horizon=1, rates=[1, 2], config=config
        )

        # Split data
        split = len(self.series) // 2
        series_train = self.series[:split]
        target_train = self.target[:split]
        series_test = self.series[split:]
        target_test = self.target[split:]

        # Fit
        clf.fit(series_train, target_train)
        assert clf.classifier is not None
        assert clf.n_series_features_ == 1

        # Predict probabilities
        proba = clf.predict_proba(series_test)
        assert proba.shape[1] == 2  # Binary classification
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

        # Predict labels
        preds = clf.predict(series_test)
        assert np.all(np.isin(preds, [0, 1]))

        # Score
        score = clf.score(series_test, target_test)
        assert 0 <= score <= 1

    def test_timeseries_classifier_predict_last(self):
        """Test predicting for most recent window."""
        arch_config = ModelArchitectureConfig(input_dim=1, n_components=16)
        train_config = TrainingConfig(max_epochs=3)
        config = CReSOConfiguration(architecture=arch_config, training=train_config)

        clf = TimeSeriesCReSOClassifier(
            window=16, horizon=1, rates=[1, 2], config=config
        )

        # Fit on part of data
        clf.fit(self.series[:-50], self.target[:-50])

        # Predict on full series (should get prediction for last window)
        pred, conf = clf.predict_last(self.series)

        assert isinstance(pred, int)
        assert pred in [0, 1]
        assert isinstance(conf, float)
        assert 0 <= conf <= 1


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="Scipy not available")
class TestGraphAdapter:
    """Test graph adapter functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        set_global_seed(42)

        # Create small test graph (two clusters)
        self.n_nodes = 20

        # Create adjacency matrix for two clusters
        adj = np.zeros((self.n_nodes, self.n_nodes))

        # Cluster 1: nodes 0-9
        for i in range(10):
            for j in range(i + 1, 10):
                if np.random.rand() > 0.3:  # Dense connections within cluster
                    adj[i, j] = adj[j, i] = 1

        # Cluster 2: nodes 10-19
        for i in range(10, 20):
            for j in range(i + 1, 20):
                if np.random.rand() > 0.3:
                    adj[i, j] = adj[j, i] = 1

        # Few connections between clusters
        adj[4, 15] = adj[15, 4] = 1
        adj[8, 12] = adj[12, 8] = 1

        self.adj = adj

        # Node features (random)
        self.X = np.random.randn(self.n_nodes, 5)

        # Labels (based on cluster membership)
        self.y = np.array([0] * 10 + [1] * 10)

        # Train/test masks
        self.train_mask = np.array([True] * 7 + [False] * 3 + [True] * 7 + [False] * 3)
        self.test_mask = ~self.train_mask

    def test_normalize_laplacian(self):
        """Test Laplacian normalization."""
        L = normalize_laplacian(self.adj)

        assert sp.issparse(L)
        assert L.shape == (self.n_nodes, self.n_nodes)

        # Should be symmetric
        assert np.allclose(L.toarray(), L.T.toarray())

        # Diagonal should be close to 1 for connected nodes
        degrees = np.sum(self.adj, axis=1)
        connected_nodes = degrees > 0
        L_diag = L.diagonal()
        assert np.all(L_diag[connected_nodes] >= 0.5)  # Reasonable lower bound

    def test_chebyshev_polynomials(self):
        """Test Chebyshev polynomial computation."""
        L = normalize_laplacian(self.adj)
        K = 3

        polynomials = chebyshev_polynomials(L, K)

        assert len(polynomials) == K

        for poly in polynomials:
            assert sp.issparse(poly)
            assert poly.shape == (self.n_nodes, self.n_nodes)

        # T_0 should be identity
        assert np.allclose(polynomials[0].toarray(), np.eye(self.n_nodes))

    def test_graph_wave_features(self):
        """Test graph wavelet feature extraction."""
        features = graph_wave_features(self.adj, self.X, K=3)

        assert features.shape == (self.n_nodes, 3 * self.X.shape[1])
        assert not np.any(np.isnan(features))

    def test_graph_wave_features_tensor_input(self):
        """Test graph features with tensor input."""
        X_tensor = torch.tensor(self.X, dtype=torch.float32)

        features = graph_wave_features(self.adj, X_tensor, K=2)

        assert isinstance(features, np.ndarray)
        assert features.shape == (self.n_nodes, 2 * self.X.shape[1])

    def test_graph_classifier_initialization(self):
        """Test graph classifier initialization."""
        arch_config = ModelArchitectureConfig(input_dim=1)
        train_config = TrainingConfig(max_epochs=3)
        config = CReSOConfiguration(architecture=arch_config, training=train_config)

        clf = GraphNodeCReSOClassifier(K=3, config=config)

        assert clf.K == 3
        assert clf.classifier is None

    def test_graph_classifier_fit_predict(self):
        """Test graph classifier fit and predict."""
        arch_config = ModelArchitectureConfig(input_dim=1, n_components=16)
        train_config = TrainingConfig(max_epochs=3)
        config = CReSOConfiguration(architecture=arch_config, training=train_config)

        clf = GraphNodeCReSOClassifier(K=3, config=config)

        # Fit with masks
        clf.fit(
            self.adj,
            self.X,
            self.y,
            train_mask=self.train_mask,
            val_mask=self.test_mask,
        )

        assert clf.classifier is not None
        assert clf.n_node_features_ == self.X.shape[1]

        # Predict probabilities
        proba = clf.predict_proba(self.adj, self.X, mask=self.test_mask)
        expected_test_nodes = np.sum(self.test_mask)
        assert proba.shape == (expected_test_nodes, 2)

        # Predict labels
        preds = clf.predict(self.adj, self.X, mask=self.test_mask)
        assert len(preds) == expected_test_nodes
        assert np.all(np.isin(preds, [0, 1]))

        # Score
        score = clf.score(self.adj, self.X, self.y, mask=self.test_mask)
        assert 0 <= score <= 1

    def test_graph_classifier_no_masks(self):
        """Test graph classifier without masks."""
        arch_config = ModelArchitectureConfig(input_dim=1, n_components=16)
        train_config = TrainingConfig(max_epochs=3)
        config = CReSOConfiguration(architecture=arch_config, training=train_config)

        clf = GraphNodeCReSOClassifier(K=2, config=config)

        # Fit without masks (use all nodes)
        clf.fit(self.adj, self.X, self.y)

        # Predict without mask
        proba = clf.predict_proba(self.adj, self.X)
        assert proba.shape == (self.n_nodes, 2)

        preds = clf.predict(self.adj, self.X)
        assert len(preds) == self.n_nodes


if __name__ == "__main__":
    pytest.main([__file__])
