"""
Visualization utilities for CReSO models and results.

Provides comprehensive plotting and visualization functions for model analysis,
performance monitoring, and spectral component inspection.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    import torch

from .logging import get_logger
from .exceptions import ValidationError, raise_validation_error

logger = get_logger(__name__)

# Optional matplotlib import with graceful fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set up plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    PLOTTING_AVAILABLE = True
    logger.info("Visualization module loaded with matplotlib support")

except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib not available - visualization functions will be limited")

    # Create dummy classes for type hints
    class DummyPlt:
        class Figure:
            pass

        class Axes:
            pass

    plt = DummyPlt()


def check_plotting_available() -> None:
    """Check if plotting libraries are available."""
    if not PLOTTING_AVAILABLE:
        raise ValidationError(
            "Plotting not available. Install with: pip install matplotlib seaborn"
        )


def plot_training_history(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accuracies: Optional[List[float]] = None,
    val_accuracies: Optional[List[float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot training history with loss and accuracy curves.

    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch (optional)
        train_accuracies: Training accuracies per epoch (optional)
        val_accuracies: Validation accuracies per epoch (optional)
        save_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure

    Raises:
        ValidationError: If plotting not available or invalid inputs
    """
    check_plotting_available()

    if not train_losses:
        raise_validation_error("train_losses cannot be empty")

    epochs = range(1, len(train_losses) + 1)

    # Determine subplot configuration
    has_accuracy = train_accuracies is not None
    n_subplots = 2 if has_accuracy else 1

    fig, axes = plt.subplots(1, n_subplots, figsize=figsize)
    if n_subplots == 1:
        axes = [axes]

    # Plot losses
    axes[0].plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    if val_losses:
        axes[0].plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracies if available
    if has_accuracy:
        axes[1].plot(
            epochs, train_accuracies, "b-", label="Training Accuracy", linewidth=2
        )
        if val_accuracies:
            axes[1].plot(
                epochs, val_accuracies, "r-", label="Validation Accuracy", linewidth=2
            )

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Training history plot saved to {save_path}")

    return fig


def plot_spectral_components(
    model: torch.nn.Module,
    top_k: int = 20,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """Plot spectral component analysis.

    Args:
        model: CReSO model with spectral components
        top_k: Number of top components to display
        save_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure

    Raises:
        ValidationError: If model doesn't have spectral components
    """
    check_plotting_available()

    # Get spectral information
    try:
        spectral_info = model.get_spectral_info()
    except AttributeError:
        raise_validation_error("Model does not have get_spectral_info method")

    # Extract data
    amp_magnitudes = spectral_info["amp_magnitudes"].cpu().numpy()
    freq_magnitudes = spectral_info["freq_magnitudes"].cpu().numpy()
    cos_amps = spectral_info["cos_amps"].cpu().numpy()
    sin_amps = spectral_info["sin_amps"].cpu().numpy()

    # Get top components by amplitude
    top_indices = np.argsort(amp_magnitudes)[-top_k:][::-1]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Amplitude distribution
    axes[0, 0].hist(
        amp_magnitudes, bins=30, alpha=0.7, color="skyblue", edgecolor="black"
    )
    axes[0, 0].set_xlabel("Amplitude Magnitude")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Distribution of Amplitude Magnitudes")
    axes[0, 0].axvline(
        np.mean(amp_magnitudes),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(amp_magnitudes):.4f}",
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Frequency distribution
    axes[0, 1].hist(
        freq_magnitudes, bins=30, alpha=0.7, color="lightcoral", edgecolor="black"
    )
    axes[0, 1].set_xlabel("Frequency Magnitude")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Distribution of Frequency Magnitudes")
    axes[0, 1].axvline(
        np.mean(freq_magnitudes),
        color="blue",
        linestyle="--",
        label=f"Mean: {np.mean(freq_magnitudes):.4f}",
    )
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Top components by amplitude
    top_amps = amp_magnitudes[top_indices]
    [f"C{i}" for i in top_indices]

    bars = axes[1, 0].bar(range(len(top_amps)), top_amps, color="gold", alpha=0.8)
    axes[1, 0].set_xlabel("Component Rank")
    axes[1, 0].set_ylabel("Amplitude Magnitude")
    axes[1, 0].set_title(f"Top {top_k} Components by Amplitude")
    axes[1, 0].set_xticks(range(0, len(top_amps), max(1, len(top_amps) // 10)))
    axes[1, 0].grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, amp) in enumerate(zip(bars, top_amps)):
        if i % max(1, len(top_amps) // 10) == 0:  # Show every nth label
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(top_amps),
                f"{amp:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # 4. Cosine vs Sine amplitude scatter
    scatter = axes[1, 1].scatter(
        cos_amps, sin_amps, c=amp_magnitudes, cmap="viridis", alpha=0.7, s=30
    )
    axes[1, 1].set_xlabel("Cosine Amplitude")
    axes[1, 1].set_ylabel("Sine Amplitude")
    axes[1, 1].set_title("Cosine vs Sine Amplitudes")
    axes[1, 1].grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label("Total Amplitude")

    # Add diagonal line for reference
    max_range = max(np.max(np.abs(cos_amps)), np.max(np.abs(sin_amps)))
    axes[1, 1].plot(
        [-max_range, max_range],
        [-max_range, max_range],
        "k--",
        alpha=0.5,
        label="Equal amplitude line",
    )
    axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Spectral analysis plot saved to {save_path}")

    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot confusion matrix with annotations.

    Args:
        confusion_matrix: Confusion matrix array
        class_names: Class names for labels
        normalize: Whether to normalize values
        save_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    check_plotting_available()

    if normalize:
        cm = (
            confusion_matrix.astype("float")
            / confusion_matrix.sum(axis=1)[:, np.newaxis]
        )
        title = "Normalized Confusion Matrix"
        fmt = ".2f"
    else:
        cm = confusion_matrix
        title = "Confusion Matrix"
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    # Set labels
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix plot saved to {save_path}")

    return fig


def plot_performance_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ["accuracy", "precision", "recall", "f1_score"],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot performance comparison across multiple models/configurations.

    Args:
        results: Dictionary of {model_name: {metric: value}}
        metrics: List of metrics to compare
        save_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    check_plotting_available()

    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    # Prepare data
    data = np.zeros((n_models, n_metrics))
    for i, model_name in enumerate(model_names):
        for j, metric in enumerate(metrics):
            data[i, j] = results[model_name].get(metric, 0.0)

    fig, ax = plt.subplots(figsize=figsize)

    # Create grouped bar chart
    x = np.arange(n_models)
    width = 0.8 / n_metrics

    colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))

    for j, metric in enumerate(metrics):
        offset = (j - n_metrics / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            data[:, j],
            width,
            label=metric.title(),
            color=colors[j],
            alpha=0.8,
        )

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Models")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Performance comparison plot saved to {save_path}")

    return fig


def plot_time_series_classification(
    series: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: Optional[np.ndarray] = None,
    predicted_probs: Optional[np.ndarray] = None,
    window_indices: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 8),
) -> plt.Figure:
    """Plot time series with classification results.

    Args:
        series: Time series data
        true_labels: True classification labels
        predicted_labels: Predicted labels (optional)
        predicted_probs: Predicted probabilities (optional)
        window_indices: Indices of classification windows (optional)
        save_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    check_plotting_available()

    fig, axes = plt.subplots(
        3 if predicted_probs is not None else 2, 1, figsize=figsize, sharex=True
    )

    t = np.arange(len(series))

    # 1. Time series with true labels
    axes[0].plot(t, series, "b-", alpha=0.7, linewidth=1)
    axes[0].set_ylabel("Series Value")
    axes[0].set_title("Time Series with Classification Labels")

    # Color background by true labels
    for i in range(len(true_labels)):
        color = "lightcoral" if true_labels[i] == 1 else "lightblue"
        if i < len(true_labels) - 1:
            axes[0].axvspan(i, i + 1, alpha=0.3, color=color)

    axes[0].grid(True, alpha=0.3)

    # 2. Classification comparison
    axes[1].plot(
        t[: len(true_labels)],
        true_labels,
        "g-",
        label="True Labels",
        linewidth=2,
        alpha=0.8,
    )

    if predicted_labels is not None:
        if window_indices is not None:
            axes[1].plot(
                window_indices,
                predicted_labels,
                "r--",
                label="Predicted Labels",
                linewidth=2,
            )
        else:
            axes[1].plot(
                t[: len(predicted_labels)],
                predicted_labels,
                "r--",
                label="Predicted Labels",
                linewidth=2,
            )

    axes[1].set_ylabel("Class Label")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. Prediction probabilities (if available)
    if predicted_probs is not None:
        if window_indices is not None:
            axes[2].plot(
                window_indices,
                predicted_probs,
                "m-",
                label="Prediction Confidence",
                linewidth=2,
            )
        else:
            axes[2].plot(
                t[: len(predicted_probs)],
                predicted_probs,
                "m-",
                label="Prediction Confidence",
                linewidth=2,
            )

        axes[2].axhline(
            y=0.5, color="k", linestyle=":", alpha=0.5, label="Decision Threshold"
        )
        axes[2].set_ylabel("Probability")
        axes[2].set_ylim(0, 1)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Time series classification plot saved to {save_path}")

    return fig


def plot_model_architecture_summary(
    model_info: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot model architecture and complexity summary.

    Args:
        model_info: Dictionary with model information
        save_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    check_plotting_available()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Parameter distribution (if available)
    if "layer_params" in model_info:
        layer_names = list(model_info["layer_params"].keys())
        param_counts = list(model_info["layer_params"].values())

        axes[0, 0].pie(
            param_counts, labels=layer_names, autopct="%1.1f%%", startangle=90
        )
        axes[0, 0].set_title("Parameter Distribution by Layer")
    else:
        axes[0, 0].text(
            0.5,
            0.5,
            "Layer parameter\ninfo not available",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
        )
        axes[0, 0].set_title("Parameter Distribution")

    # 2. Model complexity metrics
    complexity_metrics = ["Parameters", "Model Size (MB)", "Active Components"]
    complexity_values = [
        model_info.get("n_parameters", 0),
        model_info.get("model_size_mb", 0),
        model_info.get("active_components", 0),
    ]

    bars = axes[0, 1].bar(
        range(len(complexity_metrics)),
        complexity_values,
        color=["skyblue", "lightcoral", "gold"],
    )
    axes[0, 1].set_xticks(range(len(complexity_metrics)))
    axes[0, 1].set_xticklabels(complexity_metrics, rotation=45, ha="right")
    axes[0, 1].set_title("Model Complexity Metrics")

    # Add value labels
    for bar, value in zip(bars, complexity_values):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(complexity_values) * 0.01,
            f"{value}",
            ha="center",
            va="bottom",
        )

    # 3. Performance metrics (if available)
    if "performance" in model_info:
        perf_metrics = ["Accuracy", "Precision", "Recall", "F1"]
        perf_values = [
            model_info["performance"].get("accuracy", 0),
            model_info["performance"].get("precision", 0),
            model_info["performance"].get("recall", 0),
            model_info["performance"].get("f1_score", 0),
        ]

        bars = axes[1, 0].bar(
            range(len(perf_metrics)),
            perf_values,
            color=["green", "blue", "orange", "red"],
            alpha=0.7,
        )
        axes[1, 0].set_xticks(range(len(perf_metrics)))
        axes[1, 0].set_xticklabels(perf_metrics)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_title("Performance Metrics")

        # Add value labels
        for bar, value in zip(bars, perf_values):
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "Performance\nmetrics not available",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_title("Performance Metrics")

    # 4. Training summary (if available)
    if "training_history" in model_info:
        epochs = range(1, len(model_info["training_history"]["loss"]) + 1)
        axes[1, 1].plot(
            epochs,
            model_info["training_history"]["loss"],
            "b-",
            label="Loss",
            linewidth=2,
        )
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].set_title("Training Loss")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Training history\nnot available",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_title("Training History")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Model architecture summary saved to {save_path}")

    return fig


def save_all_plots(
    plots_dict: Dict[str, plt.Figure],
    output_dir: Union[str, Path],
    format: str = "png",
    dpi: int = 300,
    close_after_save: bool = True,
) -> None:
    """Save multiple plots to directory.

    Args:
        plots_dict: Dictionary of {filename: figure}
        output_dir: Output directory
        format: File format (png, pdf, svg)
        dpi: Resolution for raster formats
        close_after_save: Whether to close figures after saving
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, fig in plots_dict.items():
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"

        filepath = output_dir / filename
        fig.savefig(filepath, format=format, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved plot: {filepath}")

        if close_after_save:
            plt.close(fig)

    logger.info(f"All plots saved to {output_dir}")


def close_all_figures() -> None:
    """Close all matplotlib figures to prevent memory leaks."""
    plt.close("all")
