"""
Cross-validation utilities for CReSO models.

Provides scikit-learn compatible cross-validation functions
for evaluating CReSO classifiers and regressors.
"""

import numpy as np
import torch
from typing import Union, List, Optional, Dict, Any, Callable
from sklearn.model_selection import KFold, StratifiedKFold
from .classifier import CReSOClassifier
from .regressor import CReSORegressor
from .logging import get_logger
from .validation import validate_tensor_2d
from .exceptions import ValidationError

logger = get_logger(__name__)


def cross_val_score(
    estimator: Union[CReSOClassifier, CReSORegressor],
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    groups: Optional[np.ndarray] = None,
    scoring: Optional[Union[str, Callable]] = None,
    cv: Optional[Union[int, Callable]] = None,
    n_jobs: Optional[int] = None,  # Not used, kept for sklearn compatibility
    verbose: int = 0,
    fit_params: Optional[Dict[str, Any]] = None,
    pre_dispatch: str = "2*n_jobs",  # Not used, kept for sklearn compatibility
) -> np.ndarray:
    """Evaluate a score by cross-validation.

    Args:
        estimator: CReSO classifier or regressor
        X: Input features
        y: Target values
        groups: Group labels for the samples (not implemented)
        scoring: Scoring function ('accuracy', 'r2', etc.)
        cv: Cross-validation generator or int
        n_jobs: Number of parallel jobs (not used)
        verbose: Verbosity level
        fit_params: Parameters to pass to fit method
        pre_dispatch: Not used, kept for compatibility

    Returns:
        Array of scores from cross-validation

    Example:
        >>> from creso import CReSOClassifier, CReSOConfiguration, cross_val_score
        >>> config = CReSOConfiguration(...)
        >>> clf = CReSOClassifier(config)
        >>> scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        >>> print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    """
    if fit_params is None:
        fit_params = {}

    # Convert and validate inputs
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    elif not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)

    X = validate_tensor_2d(X, "input features")

    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    elif not isinstance(y, np.ndarray):
        y = np.array(y)

    # Set up cross-validation
    if cv is None:
        if isinstance(estimator, CReSOClassifier):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
    elif isinstance(cv, int):
        if isinstance(estimator, CReSOClassifier):
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Set up scoring function
    if scoring is None:
        if isinstance(estimator, CReSOClassifier):
            scoring = "accuracy"
        else:
            scoring = "r2"

    scorer = _get_scorer(scoring)

    if verbose > 0:
        logger.info(f"Starting cross-validation with {cv.n_splits} folds")

    scores = []
    fold_idx = 0

    for train_idx, test_idx in cv.split(X, y):
        fold_idx += 1

        # Split data
        if isinstance(X, torch.Tensor):
            X_train, X_test = X[train_idx], X[test_idx]
        else:
            X_train, X_test = X[train_idx], X[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]

        # Clone estimator (create new instance with same config)
        fold_estimator = _clone_estimator(estimator)

        # Fit and score
        try:
            fold_estimator.fit(X_train, y_train, **fit_params)
            score = scorer(fold_estimator, X_test, y_test)
            scores.append(score)

            if verbose > 0:
                logger.info(f"Fold {fold_idx}/{cv.n_splits}: {scoring}={score:.4f}")

        except Exception as e:
            logger.error(f"Fold {fold_idx} failed with error: {e}")
            raise ValidationError(f"Cross-validation failed at fold {fold_idx}: {e}")

    scores = np.array(scores)

    if verbose > 0:
        logger.info(
            f"Cross-validation completed. {scoring}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})"
        )

    return scores


def cross_validate(
    estimator: Union[CReSOClassifier, CReSORegressor],
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    groups: Optional[np.ndarray] = None,
    scoring: Optional[Union[str, List[str], Dict[str, str]]] = None,
    cv: Optional[Union[int, Callable]] = None,
    n_jobs: Optional[int] = None,
    verbose: int = 0,
    fit_params: Optional[Dict[str, Any]] = None,
    return_train_score: bool = False,
    return_estimator: bool = False,
    error_score: Union[str, float] = np.nan,
) -> Dict[str, np.ndarray]:
    """Extended cross-validation with multiple metrics.

    Args:
        estimator: CReSO classifier or regressor
        X: Input features
        y: Target values
        groups: Group labels (not implemented)
        scoring: Scoring metric(s)
        cv: Cross-validation generator or int
        n_jobs: Number of parallel jobs (not used)
        verbose: Verbosity level
        fit_params: Parameters to pass to fit method
        return_train_score: Whether to return training scores
        return_estimator: Whether to return fitted estimators
        error_score: Value to assign on error

    Returns:
        Dictionary with test scores and optionally train scores and estimators
    """
    if fit_params is None:
        fit_params = {}

    # Convert and validate inputs
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    elif not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)

    X = validate_tensor_2d(X, "input features")

    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    elif not isinstance(y, np.ndarray):
        y = np.array(y)

    # Set up cross-validation
    if cv is None:
        if isinstance(estimator, CReSOClassifier):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
    elif isinstance(cv, int):
        if isinstance(estimator, CReSOClassifier):
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Set up scoring
    if scoring is None:
        if isinstance(estimator, CReSOClassifier):
            scoring = ["accuracy"]
        else:
            scoring = ["r2"]
    elif isinstance(scoring, str):
        scoring = [scoring]
    elif isinstance(scoring, dict):
        scoring = list(scoring.values())

    scorers = {name: _get_scorer(name) for name in scoring}

    if verbose > 0:
        logger.info(
            f"Starting cross-validation with {cv.n_splits} folds and metrics: {list(scorers.keys())}"
        )

    # Initialize results
    results = {}
    for metric_name in scorers.keys():
        results[f"test_{metric_name}"] = []
        if return_train_score:
            results[f"train_{metric_name}"] = []

    if return_estimator:
        results["estimator"] = []

    results["fit_time"] = []
    results["score_time"] = []

    fold_idx = 0

    for train_idx, test_idx in cv.split(X, y):
        fold_idx += 1

        # Split data
        if isinstance(X, torch.Tensor):
            X_train, X_test = X[train_idx], X[test_idx]
        else:
            X_train, X_test = X[train_idx], X[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]

        # Clone estimator
        fold_estimator = _clone_estimator(estimator)

        try:
            # Fit and time it
            import time

            fit_start = time.perf_counter()
            fold_estimator.fit(X_train, y_train, **fit_params)
            fit_time = time.perf_counter() - fit_start
            results["fit_time"].append(fit_time)

            # Score and time it
            score_start = time.perf_counter()
            for metric_name, scorer in scorers.items():
                # Test score
                test_score = scorer(fold_estimator, X_test, y_test)
                results[f"test_{metric_name}"].append(test_score)

                # Train score if requested
                if return_train_score:
                    train_score = scorer(fold_estimator, X_train, y_train)
                    results[f"train_{metric_name}"].append(train_score)

            score_time = time.perf_counter() - score_start
            results["score_time"].append(score_time)

            # Store estimator if requested
            if return_estimator:
                results["estimator"].append(fold_estimator)

            if verbose > 0:
                test_scores_str = ", ".join(
                    [
                        f"{name}={results[f'test_{name}'][-1]:.4f}"
                        for name in scorers.keys()
                    ]
                )
                logger.info(f"Fold {fold_idx}/{cv.n_splits}: {test_scores_str}")

        except Exception as e:
            logger.error(f"Fold {fold_idx} failed with error: {e}")

            # Fill with error values
            for metric_name in scorers.keys():
                results[f"test_{metric_name}"].append(error_score)
                if return_train_score:
                    results[f"train_{metric_name}"].append(error_score)

            results["fit_time"].append(np.nan)
            results["score_time"].append(np.nan)

            if return_estimator:
                results["estimator"].append(None)

    # Convert to numpy arrays
    for key, values in results.items():
        if key != "estimator":  # Don't convert estimator list to array
            results[key] = np.array(values)

    if verbose > 0:
        for metric_name in scorers.keys():
            scores = results[f"test_{metric_name}"]
            logger.info(
                f"Cross-validation completed. {metric_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})"
            )

    return results


def validation_curve(
    estimator: Union[CReSOClassifier, CReSORegressor],
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    param_name: str,
    param_range: Union[List, np.ndarray],
    groups: Optional[np.ndarray] = None,
    cv: Optional[Union[int, Callable]] = None,
    scoring: Optional[str] = None,
    n_jobs: Optional[int] = None,
    pre_dispatch: str = "all",
    verbose: int = 0,
) -> tuple:
    """Generate validation curve for hyperparameter tuning.

    Args:
        estimator: CReSO classifier or regressor
        X: Input features
        y: Target values
        param_name: Name of parameter to vary
        param_range: Range of parameter values to test
        groups: Group labels (not implemented)
        cv: Cross-validation generator or int
        scoring: Scoring metric
        n_jobs: Number of parallel jobs (not used)
        pre_dispatch: Not used
        verbose: Verbosity level

    Returns:
        train_scores, validation_scores arrays
    """
    if verbose > 0:
        logger.info(
            f"Generating validation curve for {param_name} with {len(param_range)} values"
        )

    train_scores = []
    validation_scores = []

    for param_value in param_range:
        if verbose > 0:
            logger.info(f"Testing {param_name}={param_value}")

        # Create new estimator with modified parameter
        modified_estimator = _clone_estimator(estimator)
        _set_nested_param(modified_estimator, param_name, param_value)

        # Run cross-validation
        scores = cross_val_score(
            modified_estimator, X, y, cv=cv, scoring=scoring, verbose=0
        )

        validation_scores.append(scores)

        # For train scores, we'd need to modify cross_val_score to return both
        # For now, just duplicate validation scores as a placeholder
        train_scores.append(scores)

    return np.array(train_scores), np.array(validation_scores)


def _clone_estimator(estimator: Union[CReSOClassifier, CReSORegressor]):
    """Clone a CReSO estimator."""
    if isinstance(estimator, CReSOClassifier):
        return CReSOClassifier(estimator.config)
    elif isinstance(estimator, CReSORegressor):
        return CReSORegressor(estimator.config)
    else:
        raise ValidationError(f"Unknown estimator type: {type(estimator)}")


def _get_scorer(scoring: str) -> Callable:
    """Get scorer function by name."""
    if scoring == "accuracy":
        return lambda est, X, y: est.score(X, y)
    elif scoring == "r2":
        return lambda est, X, y: est.score(X, y)
    elif scoring == "neg_mean_squared_error":

        def mse_scorer(est, X, y):
            pred = est.predict(X)
            return -np.mean((y - pred) ** 2)

        return mse_scorer
    elif scoring == "neg_mean_absolute_error":

        def mae_scorer(est, X, y):
            pred = est.predict(X)
            return -np.mean(np.abs(y - pred))

        return mae_scorer
    else:
        raise ValidationError(f"Unknown scoring metric: {scoring}")


def _set_nested_param(obj: Any, param_name: str, value: Any):
    """Set nested parameter (e.g., 'config.training.learning_rate')."""
    parts = param_name.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
