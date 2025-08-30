"""
CReSO: Coherent Resonant Spectral Operator

A lightweight, production-friendly PyTorch implementation of spectral neural networks
that combine learnable spectral representations with geometric processing paths.
"""

from .version import __version__
from .config import CReSOConfig, CReSOConfiguration, WavePhysicsConfig
from .model import CReSOModel
from .wave_model import CReSOWaveModel
from .classifier import CReSOClassifier, CReSOvRClassifier
from .regressor import CReSORegressor
from .trainer import CReSOTrainer
from .metrics import calculate_classification_metrics, calculate_model_complexity
from .visualization import plot_training_history, plot_spectral_components
from .logging import get_logger
from .cross_validation import cross_val_score, cross_validate, validation_curve

__all__ = [
    "__version__",
    "CReSOConfig",
    "CReSOConfiguration",
    "WavePhysicsConfig",
    "CReSOModel",
    "CReSOWaveModel",
    "CReSOClassifier",
    "CReSOvRClassifier",
    "CReSORegressor",
    "CReSOTrainer",
    "calculate_classification_metrics",
    "calculate_model_complexity",
    "plot_training_history",
    "plot_spectral_components",
    "get_logger",
    "cross_val_score",
    "cross_validate",
    "validation_curve",
]
