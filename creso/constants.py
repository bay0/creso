"""
Constants and configuration values for CReSO package.

Centralizes magic numbers, default values, and package-wide constants
to improve maintainability and consistency.
"""

from __future__ import annotations

import math
from typing import Dict

# Package metadata
PACKAGE_NAME = "creso"
PACKAGE_DESCRIPTION = (
    "Coherent Resonant Spectral Operator (CReSO) - Spectral neural networks"
)
AUTHOR = "CReSO Authors"
LICENSE = "Apache-2.0"

# Model architecture defaults
DEFAULT_INPUT_DIM = 10
DEFAULT_N_COMPONENTS = 128
DEFAULT_N_GEOMETRIC_HIDDEN = 64
DEFAULT_LOCALIZED = True
DEFAULT_LEARN_CENTERS = True
DEFAULT_INIT_FREQUENCY_SCALE = 3.0

# Training defaults
DEFAULT_LEARNING_RATE = 3e-3
DEFAULT_MAX_EPOCHS = 25
DEFAULT_BATCH_SIZE = 256
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_FREQUENCY_DROPOUT = 0.2
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_USE_AMP = True
DEFAULT_EARLY_STOPPING_PATIENCE = 6

# Regularization defaults
DEFAULT_L2_FREQUENCY = 1e-4
DEFAULT_GROUP_L1 = 1e-3
DEFAULT_CENTER_DISPERSION = 1e-5

# Frequency seeding defaults
DEFAULT_USE_SEED_FREQUENCIES = True
DEFAULT_SEED_FRACTION = 0.2
DEFAULT_MAX_ACF_LAG = 12

# Numerical stability constants
EPSILON = 1e-6
MIN_STANDARD_DEVIATION = 1e-6
MIN_AMPLITUDE_THRESHOLD = 1e-6
MIN_SIGMA_VALUE = 1e-4

# Mathematical constants
PI = math.pi
TWO_PI = 2 * math.pi
SQRT_TWO_PI = math.sqrt(TWO_PI)

# Device and precision
DEFAULT_DEVICE = "cuda"  # Will fallback to "cpu" if CUDA unavailable
DEFAULT_DTYPE = "float32"
SUPPORTED_DTYPES = ["float16", "float32", "float64"]

# File formats and extensions
SUPPORTED_MODEL_EXTENSIONS = [".pkl", ".pt", ".pth"]
SUPPORTED_DATA_EXTENSIONS = [".npz", ".csv", ".json"]
SUPPORTED_EXPORT_FORMATS = ["torchscript", "onnx"]

# Validation limits
MAX_INPUT_DIMENSIONS = 10000
MAX_N_COMPONENTS = 1000
MAX_BATCH_SIZE = 100000
MAX_EPOCHS = 1000
MIN_SAMPLES_FOR_TRAINING = 10

# Time-series adapter defaults
DEFAULT_TIME_WINDOW = 128
DEFAULT_TIME_HORIZON = 1
DEFAULT_TIME_RATES = [1, 2, 4]
DEFAULT_TIME_STEP = 1
MAX_TIME_SERIES_LENGTH = 1000000

# Graph adapter defaults
DEFAULT_CHEBYSHEV_K = 3
MAX_GRAPH_NODES = 100000
MAX_CHEBYSHEV_K = 10

# Logging configuration
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "standard"
LOG_FORMATS = ["standard", "detailed", "structured"]

# Performance configuration
DEFAULT_NUM_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_PIN_MEMORY = True

# Model export configuration
DEFAULT_TORCHSCRIPT_OPTIMIZE = True
DEFAULT_ONNX_OPSET = 13
SUPPORTED_ONNX_OPSETS = [11, 12, 13, 14, 15]

# Classification thresholds
DEFAULT_BINARY_THRESHOLD = 0.5
DEFAULT_CONFIDENCE_THRESHOLD = 0.8

# Data preprocessing
DEFAULT_STANDARDIZE = True
DEFAULT_NORMALIZE = False

# Configuration presets
CONFIGURATION_PRESETS: Dict[str, Dict] = {
    "fast": {
        "n_components": 32,
        "max_epochs": 10,
        "batch_size": 512,
        "learning_rate": 1e-2,
        "use_amp": True,
        "frequency_dropout": 0.1,
    },
    "balanced": {
        "n_components": 64,
        "max_epochs": 25,
        "batch_size": 256,
        "learning_rate": 3e-3,
        "use_amp": True,
        "frequency_dropout": 0.2,
    },
    "accurate": {
        "n_components": 128,
        "max_epochs": 50,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "use_amp": False,
        "frequency_dropout": 0.3,
    },
    "minimal": {
        "n_components": 16,
        "max_epochs": 5,
        "batch_size": 64,
        "learning_rate": 1e-2,
        "use_amp": False,
        "frequency_dropout": 0.0,
    },
}

# Error codes for structured error handling
ERROR_CODES = {
    "INVALID_CONFIG": "CRESO_E001",
    "INVALID_INPUT": "CRESO_E002",
    "TRAINING_FAILED": "CRESO_E003",
    "MODEL_LOAD_FAILED": "CRESO_E004",
    "MODEL_SAVE_FAILED": "CRESO_E005",
    "EXPORT_FAILED": "CRESO_E006",
    "COMPATIBILITY_ERROR": "CRESO_E007",
    "DATA_ERROR": "CRESO_E008",
    "ADAPTER_ERROR": "CRESO_E009",
    "VALIDATION_ERROR": "CRESO_E010",
}

# Status messages
STATUS_MESSAGES = {
    "INITIALIZING": "Initializing CReSO model",
    "TRAINING": "Training in progress",
    "VALIDATING": "Validating model",
    "SAVING": "Saving model",
    "LOADING": "Loading model",
    "EXPORTING": "Exporting model",
    "PREDICTING": "Making predictions",
    "COMPLETED": "Operation completed successfully",
    "FAILED": "Operation failed",
}

# Supported tasks for CLI
SUPPORTED_CLI_TASKS = [
    "tabular_binary",
    "tabular_multiclass",
    "timeseries_binary",
    "graph_nodes_binary",
]

# Model component names (for better naming consistency)
COMPONENT_NAMES = {
    "spectral_layer": "wave_resonance_layer",
    "geometric_layer": "geometric_path",
    "gating_layer": "adaptive_gate",
    "standardizer": "feature_standardizer",
}

# Metric names for evaluation
METRIC_NAMES = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "log_loss",
    "training_loss",
    "validation_loss",
]

# Default class weights for common imbalanced scenarios
DEFAULT_CLASS_WEIGHTS = {
    "balanced": "balanced",
    "slightly_imbalanced": {0: 1.0, 1: 1.5},
    "moderately_imbalanced": {0: 1.0, 1: 2.0},
    "highly_imbalanced": {0: 1.0, 1: 5.0},
}

# Warning messages
WARNING_MESSAGES = {
    "CUDA_UNAVAILABLE": "CUDA requested but not available, falling back to CPU",
    "AMP_DISABLED": "AMP disabled due to compatibility issues",
    "EARLY_STOPPING": "Training stopped early due to convergence",
    "LARGE_MODEL": "Model has many parameters, consider reducing n_components",
    "SMALL_DATASET": "Dataset is small, consider reducing model complexity",
}

# Feature importance thresholds
FEATURE_IMPORTANCE_THRESHOLDS = {
    "very_high": 0.1,
    "high": 0.05,
    "medium": 0.01,
    "low": 0.001,
    "very_low": 0.0001,
}
