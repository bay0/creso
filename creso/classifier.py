"""
Scikit-learn style classifier wrappers for CReSO models.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import LabelEncoder

from .config import (
    CReSOConfiguration,
)
from .trainer import CReSOTrainer
from .model import CReSOModel
from .logging import get_logger
from .exceptions import ValidationError

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None

logger = get_logger(__name__)


class CReSOClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn style binary classifier using CReSO.

    Args:
        config: CReSO configuration
        **config_kwargs: Override config parameters
    """

    def __init__(
        self,
        config: Optional[CReSOConfiguration] = None,
        **config_kwargs,
    ):
        if config is None:
            # Create default config with minimal architecture
            from .config import ModelArchitectureConfig

            arch_config = ModelArchitectureConfig(input_dim=1)
            config = CReSOConfiguration(architecture=arch_config, **config_kwargs)
        else:
            # For backwards compatibility, handle dict updates
            if config_kwargs:
                config_dict = config.to_dict()
                # Update nested dictionaries appropriately
                for key, value in config_kwargs.items():
                    if key in ["learning_rate", "max_epochs", "batch_size"]:
                        config_dict["training"][key] = value
                    elif key in ["input_dim", "n_components"]:
                        config_dict["architecture"][key] = value
                config = CReSOConfiguration.from_dict(config_dict)

        self.config = config
        self.model = None
        self.standardizer = None
        self.trainer = None
        self.optimizer = None
        self.training_history = None
        self.classes_ = None
        self.n_features_in_ = None
        self._is_fitted = False

        logger.debug(
            "Initialized CReSOClassifier", extra={"config_type": type(config).__name__}
        )

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        class_weight: Optional[Union[str, Dict, float]] = None,
        **fit_kwargs,
    ) -> "CReSOClassifier":
        """Fit the classifier.

        Args:
            X: Training features (N, input_dim)
            y: Training labels (N,) - should be binary (0/1)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            class_weight: Class weighting strategy
            **fit_kwargs: Additional arguments for trainer.fit()

        Returns:
            Self for chaining
        """
        # Validate input
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # Check if binary classification
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValidationError(
                f"CReSOClassifier requires binary classification, "
                f"got {len(self.classes_)} classes"
            )

        logger.info(
            "Starting classifier training",
            extra={
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "n_classes": len(self.classes_),
                "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
            },
        )

        # Ensure labels are 0/1
        if not np.array_equal(self.classes_, [0, 1]):
            # Remap to 0/1
            y = (y == self.classes_[1]).astype(int)
            if y_val is not None:
                y_val = (y_val == self.classes_[1]).astype(int)

        # Update config with correct input dimension
        self.config.architecture.input_dim = self.n_features_in_

        # Create trainer and fit
        self.trainer = CReSOTrainer(self.config)
        self.model, self.optimizer, self.standardizer, self.training_history = (
            self.trainer.fit(X, y, X_val, y_val, class_weight, **fit_kwargs)
        )

        self._is_fitted = True
        return self

    def _check_fitted(self):
        """Check if the classifier is fitted."""
        if not self._is_fitted:
            raise ValidationError("Classifier must be fitted before making predictions")

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Input features (N, d_in)

        Returns:
            Predicted probabilities (N, 2) for [class_0, class_1]
        """
        check_array(X)
        self._check_fitted()

        return self.trainer.predict_proba(self.model, X, self.standardizer)

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Input features (N, d_in)

        Returns:
            Predicted labels (N,) in original label space
        """
        proba = self.predict_proba(X)
        predictions = (proba[:, 1] >= 0.5).astype(int)

        # Map back to original classes if they weren't 0/1
        if not np.array_equal(self.classes_, [0, 1]):
            predictions = self.classes_[predictions]

        return predictions

    def decision_function(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Compute decision function (raw scores).

        Args:
            X: Input features (N, d_in)

        Returns:
            Decision scores (N,)
        """
        self._check_fitted()

        device = next(self.model.parameters()).device
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

        if self.standardizer is not None:
            X_tensor = self.standardizer.transform(X_tensor)

        self.model.eval()
        with torch.no_grad():
            z, _, _, _, _ = self.model(X_tensor, train_mode=False)
            scores = z.squeeze().cpu().numpy()

        return scores

    def score(
        self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """Return the mean accuracy on the given test data and labels.

        Args:
            X: Test features
            y: True labels

        Returns:
            Mean accuracy score
        """
        from sklearn.metrics import accuracy_score

        predictions = self.predict(X)
        return float(accuracy_score(y, predictions))

    def save(self, path: str) -> None:
        """Save classifier to file.

        Args:
            path: Save path
        """
        self._check_fitted()

        save_data = {
            "config": self.config,
            "classes_": self.classes_,
            "n_features_in_": self.n_features_in_,
        }

        # Save model with standardizer
        model_path = path.replace(".pkl", "_model.pt")
        self.model.save(model_path, self.standardizer, save_data)

        # Save classifier metadata using torch.save
        torch.save(save_data, path)

    @classmethod
    def load(cls, path: str) -> "CReSOClassifier":
        """Load classifier from file.

        Warning: Only load models from trusted sources. Loading untrusted
        models can execute arbitrary code.

        Args:
            path: Load path

        Returns:
            Loaded classifier
        """
        import warnings

        # Security warning for PyTorch loading
        warnings.warn(
            "Loading PyTorch models from untrusted sources can execute arbitrary code. "
            "Only load models from trusted sources.",
            UserWarning,
            stacklevel=2,
        )

        # Load metadata with weights_only=False for LabelEncoder support
        save_data = torch.load(
            path,
            weights_only=False,  # Need to allow objects for LabelEncoder
            map_location="cpu",
        )

        # Load model with weights_only=False for configuration objects
        model_path = path.replace(".pkl", "_model.pt")
        model, standardizer, extra = CReSOModel.load(model_path, weights_only=False)

        # Reconstruct classifier
        classifier = cls(save_data["config"])
        classifier.model = model
        classifier.standardizer = standardizer
        classifier.trainer = CReSOTrainer(save_data["config"])
        classifier.classes_ = save_data["classes_"]
        classifier.n_features_in_ = save_data["n_features_in_"]
        classifier._is_fitted = True

        return classifier

    def to_torchscript(
        self,
        filepath: str,
        example_input: Optional[torch.Tensor] = None,
        optimize: bool = True,
        quantize: bool = False,
        return_probabilities: bool = True,
    ) -> None:
        """Export classifier to TorchScript format.

        Args:
            filepath: Path to save TorchScript model
            example_input: Example input for tracing (generates random if None)
            optimize: Whether to optimize the traced model
            quantize: Whether to apply dynamic quantization
            return_probabilities: Whether to return probabilities or predictions
        """
        self._check_fitted()

        if example_input is None:
            example_input = torch.randn(1, self.n_features_in_)

        if return_probabilities:
            # Export predict_proba method
            class CReSOClassifierProbWrapper(nn.Module):
                def __init__(self, classifier):
                    super().__init__()
                    self.classifier = classifier

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    self.classifier.model.eval()
                    device = next(self.classifier.model.parameters()).device
                    x = x.to(device)

                    if self.classifier.standardizer is not None:
                        x = self.classifier.standardizer.transform(x)

                    with torch.no_grad():
                        z, _, _, _, _ = self.classifier.model(x, train_mode=False)
                        probs = torch.sigmoid(z)
                        # Return both class probabilities
                        return torch.cat([1 - probs, probs], dim=1)

        else:
            # Export predict method
            class CReSOClassifierWrapper(nn.Module):
                def __init__(self, classifier):
                    super().__init__()
                    self.classifier = classifier

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    self.classifier.model.eval()
                    device = next(self.classifier.model.parameters()).device
                    x = x.to(device)

                    if self.classifier.standardizer is not None:
                        x = self.classifier.standardizer.transform(x)

                    with torch.no_grad():
                        z, _, _, _, _ = self.classifier.model(x, train_mode=False)
                        predictions = (torch.sigmoid(z) > 0.5).float()
                        return predictions

        # Use the model's export method
        self.model.to_torchscript(
            filepath,
            example_input,
            optimize=optimize,
            quantize=quantize,
            return_all_outputs=False,
        )

        logger.info(f"Exported CReSOClassifier to TorchScript: {filepath}")

    def to_onnx(
        self,
        filepath: str,
        example_input: Optional[torch.Tensor] = None,
        opset: int = 17,
        optimize: bool = True,
        return_probabilities: bool = True,
        verify_model: bool = True,
    ) -> None:
        """Export classifier to ONNX format.

        Args:
            filepath: Path to save ONNX model
            example_input: Example input for export (generates random if None)
            opset: ONNX opset version
            optimize: Whether to optimize the ONNX model
            return_probabilities: Whether to return probabilities or predictions
            verify_model: Whether to verify the exported model
        """
        self._check_fitted()

        if example_input is None:
            example_input = torch.randn(1, self.n_features_in_)

        # Use the model's export method
        self.model.to_onnx(
            filepath,
            example_input,
            opset=opset,
            optimize=optimize,
            return_all_outputs=False,
            verify_model=verify_model,
        )

        logger.info(f"Exported CReSOClassifier to ONNX: {filepath}")


class CReSOvRClassifier(BaseEstimator, ClassifierMixin):
    """One-vs-Rest multiclass classifier using CReSO.

    Trains one binary CReSOClassifier per class.

    Args:
        config: CReSO configuration (used for all binary classifiers)
        **config_kwargs: Override config parameters
    """

    def __init__(self, config: Optional[CReSOConfiguration] = None, **config_kwargs):
        if config is None:
            from .config import ModelArchitectureConfig

            arch_config = ModelArchitectureConfig(input_dim=1)
            config = CReSOConfiguration(architecture=arch_config, **config_kwargs)
        else:
            if config_kwargs:
                config_dict = config.to_dict()
                for key, value in config_kwargs.items():
                    if key in ["learning_rate", "max_epochs", "batch_size"]:
                        config_dict["training"][key] = value
                    elif key in ["input_dim", "n_components"]:
                        config_dict["architecture"][key] = value
                config = CReSOConfiguration.from_dict(config_dict)

        self.config = config
        self.classifiers_ = {}
        self.label_encoder_ = LabelEncoder()
        self.classes_ = None
        self.n_features_in_ = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        parallel: bool = True,
        **fit_kwargs,
    ) -> "CReSOvRClassifier":
        """Fit one-vs-rest classifiers.

        Args:
            X: Training features (N, d_in)
            y: Training labels (N,)
            parallel: Whether to use parallel training (faster for multiclass)
            **fit_kwargs: Additional arguments for each binary classifier

        Returns:
            Self for chaining
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # Encode labels
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        # Update config with correct input dimension
        self.config.architecture.input_dim = self.n_features_in_

        # Train one classifier per class
        self.classifiers_ = {}

        # Use parallel training only when beneficial (large dataset + many classes)
        use_parallel = (
            parallel
            and len(self.classes_) >= 5  # At least 5 classes
            and X.shape[0] >= 1000  # At least 1000 samples
        )

        if use_parallel:
            logger.info(f"Using parallel training for {len(self.classes_)} classes")
            self._fit_parallel(X, y_encoded, **fit_kwargs)
        else:
            self._fit_sequential(X, y_encoded, **fit_kwargs)

        return self

    def _fit_sequential(
        self, X: np.ndarray, y_encoded: np.ndarray, **fit_kwargs
    ) -> None:
        """Original sequential training method."""
        for i, class_label in enumerate(self.classes_):
            logger.info("Training classifier for class %s", class_label)

            # Create binary labels (current class vs rest)
            y_binary = (y_encoded == i).astype(int)

            # Skip if class has no positive examples
            if np.sum(y_binary) == 0:
                logger.warning("Class %s has no examples, skipping", class_label)
                continue

            # Train binary classifier
            clf = CReSOClassifier(self.config)
            clf.fit(X, y_binary, **fit_kwargs)
            self.classifiers_[i] = clf

    def _fit_parallel(self, X: np.ndarray, y_encoded: np.ndarray, **fit_kwargs) -> None:
        """Optimized parallel training method."""
        import concurrent.futures
        import copy

        def train_single_classifier(args):
            """Train a single binary classifier."""
            i, class_label, X_data, y_encoded_data = args

            # Create binary labels (current class vs rest)
            y_binary = (y_encoded_data == i).astype(int)

            # Skip if class has no positive examples
            if np.sum(y_binary) == 0:
                logger.warning("Class %s has no examples, skipping", class_label)
                return i, None

            # Create a copy of config to avoid race conditions
            config_copy = copy.deepcopy(self.config)
            clf = CReSOClassifier(config_copy)
            clf.fit(X_data, y_binary, **fit_kwargs)
            return i, clf

        # Prepare arguments for parallel execution
        args_list = [
            (i, class_label, X, y_encoded)
            for i, class_label in enumerate(self.classes_)
        ]

        # Use ThreadPoolExecutor for I/O bound tasks or ProcessPoolExecutor for CPU bound
        # Using ThreadPoolExecutor for better memory efficiency
        max_workers = min(len(self.classes_), 4)  # Limit workers to avoid memory issues

        logger.info(
            f"Training {len(self.classes_)} classifiers in parallel with {max_workers} workers"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(train_single_classifier, args) for args in args_list
            ]

            for future in concurrent.futures.as_completed(futures):
                i, clf = future.result()
                if clf is not None:
                    self.classifiers_[i] = clf

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Input features (N, d_in)

        Returns:
            Predicted probabilities (N, n_classes)
        """
        check_array(X)

        if not self.classifiers_:
            raise ValidationError("Model must be fitted before prediction")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes))

        # Get predictions from each binary classifier
        for i, clf in self.classifiers_.items():
            probas[:, i] = clf.predict_proba(X)[:, 1]  # Positive class probability

        # Normalize probabilities to sum to 1
        row_sums = probas.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        probas = probas / row_sums

        return probas

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Input features (N, d_in)

        Returns:
            Predicted labels (N,) in original label space
        """
        probas = self.predict_proba(X)
        predicted_indices = np.argmax(probas, axis=1)
        return self.label_encoder_.inverse_transform(predicted_indices)

    def decision_function(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Compute decision function for each class.

        Args:
            X: Input features (N, d_in)

        Returns:
            Decision scores (N, n_classes)
        """
        check_array(X)

        if not self.classifiers_:
            raise ValidationError("Model must be fitted before prediction")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes))

        for i, clf in self.classifiers_.items():
            scores[:, i] = clf.decision_function(X)

        return scores

    def save(self, path_dir: str) -> None:
        """Save multiclass classifier to directory.

        Args:
            path_dir: Directory to save classifiers
        """
        if not self.classifiers_:
            raise ValueError("Cannot save unfitted classifier")

        os.makedirs(path_dir, exist_ok=True)

        # Save metadata
        metadata = {
            "config": self.config,
            "classes_": self.classes_,
            "n_features_in_": self.n_features_in_,
            "label_encoder_": self.label_encoder_,
        }

        torch.save(metadata, os.path.join(path_dir, "metadata.pkl"))

        # Save each binary classifier
        for i, clf in self.classifiers_.items():
            clf_path = os.path.join(path_dir, f"classifier_{i}.pkl")
            clf.save(clf_path)

    @classmethod
    def load(cls, path_dir: str) -> "CReSOvRClassifier":
        """Load multiclass classifier from directory.

        Warning: Only load models from trusted sources. Loading untrusted
        models can execute arbitrary code.

        Args:
            path_dir: Directory containing saved classifiers

        Returns:
            Loaded classifier
        """
        import warnings

        # Security warning for PyTorch loading
        warnings.warn(
            "Loading PyTorch models from untrusted sources can execute arbitrary code. "
            "Only load models from trusted sources.",
            UserWarning,
            stacklevel=2,
        )

        # Load metadata
        # Load metadata with weights_only=False for LabelEncoder support
        metadata = torch.load(
            os.path.join(path_dir, "metadata.pkl"),
            weights_only=False,  # Need to allow objects for LabelEncoder
            map_location="cpu",
        )

        # Reconstruct classifier
        classifier = cls(metadata["config"])
        classifier.classes_ = metadata["classes_"]
        classifier.n_features_in_ = metadata["n_features_in_"]
        classifier.label_encoder_ = metadata["label_encoder_"]

        # Load binary classifiers
        classifier.classifiers_ = {}
        for i in range(len(classifier.classes_)):
            clf_path = os.path.join(path_dir, f"classifier_{i}.pkl")
            if os.path.exists(clf_path):
                classifier.classifiers_[i] = CReSOClassifier.load(clf_path)

        return classifier
