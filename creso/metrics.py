"""
Evaluation metrics and performance monitoring for CReSO models.

Provides comprehensive metrics calculation, model performance analysis,
and statistical utilities for enterprise-grade monitoring.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import time
import warnings

from .logging import get_logger
from .exceptions import ValidationError, raise_validation_error

logger = get_logger(__name__)


@dataclass
class ClassificationMetrics:
    """Container for classification performance metrics.

    Args:
        accuracy: Overall accuracy score
        precision: Precision score (macro-averaged for multiclass)
        recall: Recall score (macro-averaged for multiclass)
        f1_score: F1 score (macro-averaged for multiclass)
        roc_auc: ROC AUC score (for binary classification)
        log_loss: Logarithmic loss
        confusion_matrix: Confusion matrix
        classification_report: Detailed classification report
        n_samples: Number of samples evaluated
        n_classes: Number of classes
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    log_loss: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]
    n_samples: int
    n_classes: int

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"Classification Metrics (n={self.n_samples}, classes={self.n_classes})",
            f"  Accuracy:  {self.accuracy:.4f}",
            f"  Precision: {self.precision:.4f}",
            f"  Recall:    {self.recall:.4f}",
            f"  F1 Score:  {self.f1_score:.4f}",
            f"  Log Loss:  {self.log_loss:.4f}",
        ]

        if self.roc_auc is not None:
            lines.append(f"  ROC AUC:   {self.roc_auc:.4f}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/serialization."""
        return {
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1_score": float(self.f1_score),
            "roc_auc": float(self.roc_auc) if self.roc_auc is not None else None,
            "log_loss": float(self.log_loss),
            "n_samples": int(self.n_samples),
            "n_classes": int(self.n_classes),
        }


@dataclass
class ModelComplexityMetrics:
    """Container for model complexity metrics.

    Args:
        n_parameters: Total number of parameters
        n_trainable_parameters: Number of trainable parameters
        model_size_mb: Model size in megabytes
        memory_usage_mb: Memory usage during inference (MB)
        inference_time_ms: Average inference time (milliseconds)
        flops: Floating point operations for forward pass
        active_spectral_components: Number of active spectral components
        spectral_sparsity: Fraction of pruned spectral components
    """

    n_parameters: int
    n_trainable_parameters: int
    model_size_mb: float
    memory_usage_mb: Optional[float]
    inference_time_ms: Optional[float]
    flops: Optional[int]
    active_spectral_components: Optional[int]
    spectral_sparsity: Optional[float]

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            "Model Complexity Metrics",
            f"  Parameters: {self.n_parameters:,} ({self.n_trainable_parameters:,} trainable)",
            f"  Model Size: {self.model_size_mb:.2f} MB",
        ]

        if self.memory_usage_mb:
            lines.append(f"  Memory Usage: {self.memory_usage_mb:.2f} MB")

        if self.inference_time_ms:
            lines.append(f"  Inference Time: {self.inference_time_ms:.2f} ms")

        if self.active_spectral_components:
            lines.append(
                f"  Active Spectral Components: {self.active_spectral_components}"
            )

        if self.spectral_sparsity:
            lines.append(f"  Spectral Sparsity: {self.spectral_sparsity:.2%}")

        return "\n".join(lines)


def calculate_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
    class_names: Optional[List[str]] = None,
    average: str = "macro",
) -> ClassificationMetrics:
    """Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for ROC AUC and log loss)
        class_names: Names of classes for reporting
        average: Averaging strategy for multiclass metrics

    Returns:
        Classification metrics container

    Raises:
        ValidationError: If inputs are invalid
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            log_loss,
            confusion_matrix,
            classification_report,
        )
    except ImportError:
        raise ValidationError("sklearn required for metrics calculation")

    # Convert to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_prob is not None and isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()

    # Validate inputs
    if len(y_true) != len(y_pred):
        raise_validation_error(
            "y_true and y_pred must have same length",
            context={"y_true_len": len(y_true), "y_pred_len": len(y_pred)},
        )

    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate classification report
    target_names = class_names if class_names else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
        )

    # Calculate ROC AUC for binary classification
    roc_auc = None
    if y_prob is not None:
        n_classes = len(np.unique(y_true))
        try:
            if n_classes == 2:
                # Binary classification
                roc_auc = roc_auc_score(
                    y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                )
            else:
                # Multiclass classification
                roc_auc = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average=average
                )
        except ValueError as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")

    # Calculate log loss
    log_loss_val = 0.0
    if y_prob is not None:
        try:
            log_loss_val = log_loss(y_true, y_prob)
        except ValueError as e:
            logger.warning(f"Could not calculate log loss: {e}")

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        roc_auc=roc_auc,
        log_loss=log_loss_val,
        confusion_matrix=cm,
        classification_report=report,
        n_samples=len(y_true),
        n_classes=len(np.unique(y_true)),
    )


def calculate_model_complexity(model: torch.nn.Module) -> ModelComplexityMetrics:
    """Calculate model complexity metrics.

    Args:
        model: PyTorch model to analyze

    Returns:
        Model complexity metrics
    """
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)

    # Try to get spectral component information
    active_components = None
    spectral_sparsity = None

    if hasattr(model, "wave_layer") or hasattr(model, "wave_resonance_layer"):
        try:
            layer = getattr(
                model, "wave_layer", getattr(model, "wave_resonance_layer", None)
            )
            if layer and hasattr(layer, "get_amplitude_magnitudes"):
                amps = layer.get_amplitude_magnitudes()
                active_components = torch.sum(amps > 1e-6).item()
                total_components = len(amps)
                spectral_sparsity = 1.0 - (active_components / total_components)
        except Exception as e:
            logger.debug(f"Could not analyze spectral components: {e}")

    return ModelComplexityMetrics(
        n_parameters=n_params,
        n_trainable_parameters=n_trainable,
        model_size_mb=model_size_mb,
        memory_usage_mb=None,
        inference_time_ms=None,
        flops=None,
        active_spectral_components=active_components,
        spectral_sparsity=spectral_sparsity,
    )


def benchmark_inference_time(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    n_runs: int = 100,
    warmup_runs: int = 10,
) -> Tuple[float, float]:
    """Benchmark model inference time.

    Args:
        model: Model to benchmark
        sample_input: Sample input tensor
        n_runs: Number of timing runs
        warmup_runs: Number of warmup runs

    Returns:
        (mean_time_ms, std_time_ms) tuple
    """
    model.eval()
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)

    # Timing runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            _ = model(sample_input)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms

    return np.mean(times), np.std(times)


def calculate_spectral_statistics(model: torch.nn.Module) -> Dict[str, Any]:
    """Calculate statistics about spectral components.

    Args:
        model: CReSO model to analyze

    Returns:
        Dictionary of spectral statistics
    """
    stats = {}

    # Try to access spectral layer
    spectral_layer = None
    if hasattr(model, "wave_layer"):
        spectral_layer = model.wave_layer
    elif hasattr(model, "wave_resonance_layer"):
        spectral_layer = model.wave_resonance_layer

    if spectral_layer is None:
        logger.warning("No spectral layer found in model")
        return stats

    try:
        # Get amplitude and frequency information
        if hasattr(spectral_layer, "get_amplitude_magnitudes"):
            amp_mags = spectral_layer.get_amplitude_magnitudes().detach().cpu().numpy()
            stats["amplitude_statistics"] = {
                "mean": float(np.mean(amp_mags)),
                "std": float(np.std(amp_mags)),
                "min": float(np.min(amp_mags)),
                "max": float(np.max(amp_mags)),
                "median": float(np.median(amp_mags)),
            }

            # Active component analysis
            active_threshold = 1e-6
            active_mask = amp_mags > active_threshold
            stats["active_components"] = {
                "count": int(np.sum(active_mask)),
                "fraction": float(np.mean(active_mask)),
                "threshold": active_threshold,
            }

        if hasattr(spectral_layer, "get_frequency_magnitudes"):
            freq_mags = spectral_layer.get_frequency_magnitudes().detach().cpu().numpy()
            stats["frequency_statistics"] = {
                "mean": float(np.mean(freq_mags)),
                "std": float(np.std(freq_mags)),
                "min": float(np.min(freq_mags)),
                "max": float(np.max(freq_mags)),
                "median": float(np.median(freq_mags)),
            }

        # Component distribution analysis
        if hasattr(spectral_layer, "a_c") and hasattr(spectral_layer, "a_s"):
            cos_amps = spectral_layer.a_c.detach().cpu().numpy()
            sin_amps = spectral_layer.a_s.detach().cpu().numpy()

            stats["component_balance"] = {
                "cos_mean": float(np.mean(np.abs(cos_amps))),
                "sin_mean": float(np.mean(np.abs(sin_amps))),
                "cos_std": float(np.std(cos_amps)),
                "sin_std": float(np.std(sin_amps)),
            }

    except Exception as e:
        logger.warning(f"Error calculating spectral statistics: {e}")

    return stats


def create_performance_summary(
    classification_metrics: ClassificationMetrics,
    model_complexity: ModelComplexityMetrics,
    spectral_stats: Optional[Dict[str, Any]] = None,
    training_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Create comprehensive performance summary.

    Args:
        classification_metrics: Classification performance metrics
        model_complexity: Model complexity metrics
        spectral_stats: Spectral component statistics
        training_time: Training time in seconds

    Returns:
        Comprehensive performance summary
    """
    summary = {
        "performance": classification_metrics.to_dict(),
        "complexity": {
            "n_parameters": model_complexity.n_parameters,
            "model_size_mb": model_complexity.model_size_mb,
            "active_components": model_complexity.active_spectral_components,
            "spectral_sparsity": model_complexity.spectral_sparsity,
        },
        "efficiency": {},
    }

    if model_complexity.inference_time_ms:
        summary["efficiency"]["inference_time_ms"] = model_complexity.inference_time_ms

        # Calculate throughput
        samples_per_second = 1000 / model_complexity.inference_time_ms
        summary["efficiency"]["throughput_samples_per_sec"] = samples_per_second

    if training_time:
        summary["efficiency"]["training_time_sec"] = training_time
        summary["efficiency"]["samples_per_training_sec"] = (
            classification_metrics.n_samples / training_time
        )

    if spectral_stats:
        summary["spectral_analysis"] = spectral_stats

    # Calculate overall efficiency score
    efficiency_score = calculate_efficiency_score(
        classification_metrics, model_complexity
    )
    summary["efficiency_score"] = efficiency_score

    return summary


def calculate_efficiency_score(
    metrics: ClassificationMetrics, complexity: ModelComplexityMetrics
) -> float:
    """Calculate overall efficiency score balancing accuracy and complexity.

    Args:
        metrics: Classification metrics
        complexity: Model complexity metrics

    Returns:
        Efficiency score (higher is better)
    """
    # Base score from accuracy
    accuracy_score = metrics.accuracy

    # Penalty for model complexity (normalize by typical values)
    param_penalty = min(complexity.n_parameters / 100000, 1.0) * 0.1
    size_penalty = min(complexity.model_size_mb / 100, 1.0) * 0.1

    # Bonus for spectral sparsity
    sparsity_bonus = 0.0
    if complexity.spectral_sparsity:
        sparsity_bonus = complexity.spectral_sparsity * 0.1

    efficiency = accuracy_score - param_penalty - size_penalty + sparsity_bonus

    return max(0.0, min(1.0, efficiency))
