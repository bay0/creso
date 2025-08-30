"""
Graph adapter for CReSO models.

Provides Chebyshev wavelet features for graph node classification.
"""

import numpy as np
import torch
from typing import Optional, Union

try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..classifier import CReSOClassifier
from ..config import CReSOConfig
from ..logging import get_logger

logger = get_logger(__name__)


def normalize_laplacian(adj: Union[np.ndarray, "sp.csr_matrix"]) -> "sp.csr_matrix":
    """Compute normalized graph Laplacian.

    L = I - D^(-1/2) A D^(-1/2)

    Args:
        adj: Adjacency matrix (N, N)

    Returns:
        Normalized Laplacian matrix
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("Graph adapter requires scipy: pip install scipy")

    # Convert to sparse if needed
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    # Compute degree matrix
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1  # Avoid division by zero for isolated nodes

    # D^(-1/2)
    d_inv_sqrt = sp.diags(np.power(degrees, -0.5))

    # Normalized adjacency: D^(-1/2) A D^(-1/2)
    norm_adj = d_inv_sqrt @ adj @ d_inv_sqrt

    # Laplacian: I - norm_adj
    n = adj.shape[0]
    identity = sp.eye(n, format="csr")
    laplacian = identity - norm_adj

    return laplacian


def chebyshev_polynomials(L: "sp.csr_matrix", K: int = 3) -> list:
    """Compute Chebyshev polynomials of the Laplacian.

    T_0(L) = I
    T_1(L) = L
    T_k(L) = 2 * L * T_{k-1}(L) - T_{k-2}(L)

    Args:
        L: Normalized Laplacian matrix
        K: Number of Chebyshev polynomials

    Returns:
        List of Chebyshev polynomial matrices
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("Graph adapter requires scipy: pip install scipy")

    n = L.shape[0]

    # Scale Laplacian to [-1, 1] range for Chebyshev polynomials
    # Estimate largest eigenvalue
    try:
        lambda_max = eigsh(L, k=1, which="LA", return_eigenvectors=False)[0]
    except Exception:
        # Fallback: use 2 as approximation
        lambda_max = 2.0

    L_scaled = (2.0 / lambda_max) * L - sp.eye(n, format="csr")

    # Compute Chebyshev polynomials
    polynomials = []

    if K >= 1:
        # T_0 = I
        polynomials.append(sp.eye(n, format="csr"))

    if K >= 2:
        # T_1 = L_scaled
        polynomials.append(L_scaled)

    # T_k = 2 * L_scaled * T_{k-1} - T_{k-2}
    for k in range(2, K):
        T_k = 2 * L_scaled @ polynomials[k - 1] - polynomials[k - 2]
        polynomials.append(T_k)

    return polynomials


def graph_wave_features(
    adj: Union[np.ndarray, "sp.csr_matrix"],
    X: Union[np.ndarray, torch.Tensor],
    K: int = 3,
) -> np.ndarray:
    """Extract graph wavelet features using Chebyshev polynomials.

    For each node, computes features by applying Chebyshev polynomials
    of the graph Laplacian to the node features.

    Args:
        adj: Adjacency matrix (N, N)
        X: Node features (N, D)
        K: Number of Chebyshev polynomials

    Returns:
        Graph wavelet features (N, K * D)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("Graph adapter requires scipy: pip install scipy")

    # Convert to numpy if needed
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    n_nodes, n_features = X.shape

    # Compute normalized Laplacian
    L = normalize_laplacian(adj)

    # Compute Chebyshev polynomials
    polynomials = chebyshev_polynomials(L, K)

    # Apply each polynomial to the features
    wave_features = []

    for poly in polynomials:
        # Apply polynomial: poly @ X
        filtered_features = poly @ X  # (N, D)
        wave_features.append(filtered_features)

    # Concatenate all filtered features
    wave_features = np.concatenate(wave_features, axis=1)  # (N, K * D)

    return wave_features


class GraphNodeCReSOClassifier:
    """Graph node classifier using CReSO with Chebyshev wavelet features.

    Automatically extracts graph spectral features using Chebyshev polynomials
    and trains a CReSO classifier for node classification.

    Args:
        K: Number of Chebyshev polynomials for wavelet features
        config: CReSO configuration (will be updated with correct d_in)
        **config_kwargs: Override config parameters
    """

    def __init__(
        self, K: int = 3, config: Optional[CReSOConfig] = None, **config_kwargs
    ):
        if not SCIPY_AVAILABLE:
            raise ImportError("Graph adapter requires scipy: pip install scipy")

        self.K = K

        # Create config (d_in will be set during fit)
        if config is None:
            config = CReSOConfig(d_in=1, **config_kwargs)
        else:
            for key, value in config_kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self.config = config
        self.classifier = None
        self.n_node_features_ = None

    def fit(
        self,
        adj: Union[np.ndarray, "sp.csr_matrix"],
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        train_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        val_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **fit_kwargs,
    ) -> "GraphNodeCReSOClassifier":
        """Fit graph node classifier.

        Args:
            adj: Adjacency matrix (N, N)
            X: Node features (N, D)
            y: Node labels (N,)
            train_mask: Boolean mask for training nodes (optional)
            val_mask: Boolean mask for validation nodes (optional)
            **fit_kwargs: Additional arguments for classifier.fit()

        Returns:
            Self for chaining
        """
        # Extract graph wavelet features
        X_wave = graph_wave_features(adj, X, self.K)

        # Track original feature dimension
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        self.n_node_features_ = X.shape[1]

        # Convert labels to numpy if needed
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        # Apply training mask if provided
        if train_mask is not None:
            if isinstance(train_mask, torch.Tensor):
                train_mask = train_mask.cpu().numpy()
            X_train = X_wave[train_mask]
            y_train = y[train_mask]
        else:
            X_train = X_wave
            y_train = y

        # Apply validation mask if provided
        X_val, y_val = None, None
        if val_mask is not None:
            if isinstance(val_mask, torch.Tensor):
                val_mask = val_mask.cpu().numpy()
            X_val = X_wave[val_mask]
            y_val = y[val_mask]

        # Update config with correct feature dimension
        self.config.d_in = X_train.shape[1]

        # Create and fit classifier
        self.classifier = CReSOClassifier(self.config)
        self.classifier.fit(X_train, y_train, X_val, y_val, **fit_kwargs)

        return self

    def predict_proba(
        self,
        adj: Union[np.ndarray, "sp.csr_matrix"],
        X: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        """Predict probabilities for graph nodes.

        Args:
            adj: Adjacency matrix (N, N)
            X: Node features (N, D)
            mask: Boolean mask for nodes to predict (optional)

        Returns:
            Predicted probabilities (N, 2) or (|mask|, 2) if mask provided
        """
        if self.classifier is None:
            raise ValueError("Classifier must be fitted before prediction")

        # Extract graph wavelet features
        X_wave = graph_wave_features(adj, X, self.K)

        # Apply mask if provided
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            X_wave = X_wave[mask]

        return self.classifier.predict_proba(X_wave)

    def predict(
        self,
        adj: Union[np.ndarray, "sp.csr_matrix"],
        X: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        """Predict labels for graph nodes.

        Args:
            adj: Adjacency matrix (N, N)
            X: Node features (N, D)
            mask: Boolean mask for nodes to predict (optional)

        Returns:
            Predicted labels (N,) or (|mask|,) if mask provided
        """
        if self.classifier is None:
            raise ValueError("Classifier must be fitted before prediction")

        X_wave = graph_wave_features(adj, X, self.K)

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            X_wave = X_wave[mask]

        return self.classifier.predict(X_wave)

    def score(
        self,
        adj: Union[np.ndarray, "sp.csr_matrix"],
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> float:
        """Compute accuracy score for graph nodes.

        Args:
            adj: Adjacency matrix (N, N)
            X: Node features (N, D)
            y: True labels (N,)
            mask: Boolean mask for nodes to evaluate (optional)

        Returns:
            Accuracy score
        """
        predictions = self.predict(adj, X, mask)

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            y = y[mask]

        return np.mean(predictions == y)

    def save(self, path: str) -> None:
        """Save graph classifier.

        Args:
            path: Save path
        """
        if self.classifier is None:
            raise ValueError("Cannot save unfitted classifier")

        # Save the underlying classifier
        self.classifier.save(path)

        # Save graph-specific parameters
        import pickle

        graph_params = {"K": self.K, "n_node_features_": self.n_node_features_}

        graph_path = path.replace(".pkl", "_graph_params.pkl")
        with open(graph_path, "wb") as f:
            pickle.dump(graph_params, f)

    @classmethod
    def load(cls, path: str) -> "GraphNodeCReSOClassifier":
        """Load graph classifier.

        Args:
            path: Load path

        Returns:
            Loaded classifier
        """
        # Load the underlying classifier
        classifier = CReSOClassifier.load(path)

        # Load graph parameters
        import pickle
        import warnings

        graph_path = path.replace(".pkl", "_graph_params.pkl")
        
        # Security warning for pickle loading
        warnings.warn(
            "Loading pickled data from untrusted sources can execute arbitrary code. "
            "Only load models from trusted sources.",
            UserWarning,
            stacklevel=2
        )
        
        with open(graph_path, "rb") as f:
            graph_params = pickle.load(f)

        # Reconstruct graph classifier
        graph_clf = cls(K=graph_params["K"], config=classifier.config)
        graph_clf.classifier = classifier
        graph_clf.n_node_features_ = graph_params["n_node_features_"]

        return graph_clf
