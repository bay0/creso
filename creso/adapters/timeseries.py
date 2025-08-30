"""
Time-series adapter for CReSO models.

Provides multi-rate windowing and time-series specific classifier.
"""

import numpy as np
import torch
from typing import List, Optional, Tuple, Union

from ..classifier import CReSOClassifier
from ..config import CReSOConfig
from ..logging import get_logger
from ..exceptions import ValidationError

logger = get_logger(__name__)


def make_multirate_windows(
    series: Union[np.ndarray, torch.Tensor],
    target: Optional[Union[np.ndarray, torch.Tensor]] = None,
    window: int = 128,
    horizon: int = 1,
    rates: List[int] = [1, 2, 4],
    step: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Create multi-rate windowed features from time series.

    Creates windows at different sampling rates and concatenates them to form
    feature vectors that capture patterns at multiple time scales.

    Args:
        series: Time series data (T,) or (T, D) for multivariate
        target: Target values (T,) for supervised learning (optional)
        window: Base window size for feature extraction
        horizon: Prediction horizon (how far ahead to predict)
        rates: List of downsampling rates (1=original, 2=every 2nd point, etc.)
        step: Step size for sliding window

    Returns:
        X: Feature matrix (N, window * len(rates) * D)
        y: Targets (N,) if target provided, else None
    """
    if isinstance(series, torch.Tensor):
        series = series.cpu().numpy()
    if target is not None and isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    # Ensure series is 2D (T, D)
    if series.ndim == 1:
        series = series.reshape(-1, 1)

    T, D = series.shape
    max_rate = max(rates)

    # We need enough data for the largest rate
    min_length = window * max_rate + horizon
    if T < min_length:
        raise ValueError(
            f"Series too short. Need at least {min_length} points, got {T}"
        )

    X_list = []
    y_list = []

    # Generate windows with sliding step
    for start in range(0, T - min_length + 1, step):
        window_features = []

        # Extract features at each rate
        for rate in rates:
            # For this rate, we need 'window' points spaced 'rate' apart
            end_idx = start + window * rate
            if end_idx <= T:
                # Extract every 'rate'-th point
                rate_window = series[start:end_idx:rate]  # (window, D)
                window_features.append(rate_window.flatten())  # (window * D,)

        if len(window_features) == len(rates):
            # All rates succeeded, concatenate features
            X_list.append(np.concatenate(window_features))  # (window * len(rates) * D,)

            # Extract target at prediction horizon
            if target is not None:
                target_idx = start + window * max_rate + horizon - 1
                if target_idx < T:
                    y_list.append(target[target_idx])

    if not X_list:
        raise ValueError("No valid windows could be created")

    X = np.array(X_list)
    y = np.array(y_list) if y_list else None

    return X, y


class TimeSeriesCReSOClassifier:
    """Time-series classifier using CReSO with multi-rate windowing.

    Automatically handles windowing and multi-rate feature extraction
    for time-series classification tasks.

    Args:
        window: Window size for feature extraction
        horizon: Prediction horizon
        rates: List of sampling rates for multi-rate analysis
        step: Step size for window sliding
        config: CReSO configuration (will be updated with correct d_in)
        **config_kwargs: Override config parameters
    """

    def __init__(
        self,
        window: int = 128,
        horizon: int = 1,
        rates: List[int] = [1, 2, 4],
        step: int = 1,
        config: Optional[CReSOConfig] = None,
        **config_kwargs,
    ):
        self.window = window
        self.horizon = horizon
        self.rates = rates
        self.step = step

        # Create config (d_in will be set during fit)
        if config is None:
            config = CReSOConfig(d_in=1, **config_kwargs)
        else:
            for key, value in config_kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self.config = config
        self.classifier = None
        self.n_series_features_ = None

    def fit(
        self,
        series: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor],
        series_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        target_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **fit_kwargs,
    ) -> "TimeSeriesCReSOClassifier":
        """Fit time-series classifier.

        Args:
            series: Training time series (T,) or (T, D)
            target: Training targets (T,)
            series_val: Validation series (optional)
            target_val: Validation targets (optional)
            **fit_kwargs: Additional arguments for classifier.fit()

        Returns:
            Self for chaining
        """
        # Create windowed features
        X_train, y_train = make_multirate_windows(
            series, target, self.window, self.horizon, self.rates, self.step
        )

        # Track number of series features for prediction
        if isinstance(series, torch.Tensor):
            series = series.cpu().numpy()
        if series.ndim == 1:
            self.n_series_features_ = 1
        else:
            self.n_series_features_ = series.shape[1]

        # Validation data - ensure both provided or both None
        if (series_val is None) != (target_val is None):
            raise ValidationError(
                "Both series_val and target_val must be provided or both must be None"
            )

        X_val, y_val = None, None
        if series_val is not None and target_val is not None:
            X_val, y_val = make_multirate_windows(
                series_val, target_val, self.window, self.horizon, self.rates, self.step
            )

        # Update config with correct feature dimension
        self.config.d_in = X_train.shape[1]

        # Create and fit classifier
        self.classifier = CReSOClassifier(self.config)
        self.classifier.fit(X_train, y_train, X_val, y_val, **fit_kwargs)

        return self

    def predict_proba(self, series: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict probabilities for time series.

        Args:
            series: Time series data (T,) or (T, D)

        Returns:
            Predicted probabilities (N, 2) where N is number of valid windows
        """
        if self.classifier is None:
            raise ValueError("Classifier must be fitted before prediction")

        # Create windowed features (no targets needed for prediction)
        X, _ = make_multirate_windows(
            series, None, self.window, self.horizon, self.rates, self.step
        )

        return self.classifier.predict_proba(X)

    def predict(self, series: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predict labels for time series.

        Args:
            series: Time series data (T,) or (T, D)

        Returns:
            Predicted labels (N,) where N is number of valid windows
        """
        if self.classifier is None:
            raise ValueError("Classifier must be fitted before prediction")

        X, _ = make_multirate_windows(
            series, None, self.window, self.horizon, self.rates, self.step
        )

        return self.classifier.predict(X)

    def predict_last(
        self, series: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[int, float]:
        """Predict for the most recent window (typical use case).

        Args:
            series: Time series data (T,) or (T, D)

        Returns:
            (prediction, confidence) for the latest window
        """
        probas = self.predict_proba(series)
        if len(probas) == 0:
            raise ValueError("No valid windows for prediction")

        # Take the last (most recent) prediction
        last_proba = probas[-1]
        prediction = int(np.argmax(last_proba))
        confidence = float(np.max(last_proba))

        return prediction, confidence

    def score(
        self,
        series: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor],
    ) -> float:
        """Compute accuracy score.

        Args:
            series: Time series data
            target: True targets

        Returns:
            Accuracy score
        """
        X, y = make_multirate_windows(
            series, target, self.window, self.horizon, self.rates, self.step
        )

        predictions = self.classifier.predict(X)
        return np.mean(predictions == y)

    def save(self, path: str) -> None:
        """Save time-series classifier.

        Args:
            path: Save path
        """
        if self.classifier is None:
            raise ValueError("Cannot save unfitted classifier")

        # Save the underlying classifier
        self.classifier.save(path)

        # Save time-series specific parameters
        import pickle

        ts_params = {
            "window": self.window,
            "horizon": self.horizon,
            "rates": self.rates,
            "step": self.step,
            "n_series_features_": self.n_series_features_,
        }

        ts_path = path.replace(".pkl", "_ts_params.pkl")
        with open(ts_path, "wb") as f:
            pickle.dump(ts_params, f)

    @classmethod
    def load(cls, path: str) -> "TimeSeriesCReSOClassifier":
        """Load time-series classifier.

        Args:
            path: Load path

        Returns:
            Loaded classifier
        """
        # Load the underlying classifier
        classifier = CReSOClassifier.load(path)

        # Load time-series parameters
        import pickle

        ts_path = path.replace(".pkl", "_ts_params.pkl")
        with open(ts_path, "rb") as f:
            ts_params = pickle.load(f)

        # Reconstruct time-series classifier
        ts_clf = cls(
            window=ts_params["window"],
            horizon=ts_params["horizon"],
            rates=ts_params["rates"],
            step=ts_params["step"],
            config=classifier.config,
        )
        ts_clf.classifier = classifier
        ts_clf.n_series_features_ = ts_params["n_series_features_"]

        return ts_clf
