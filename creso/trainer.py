"""
Training utilities for CReSO models.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from pathlib import Path
from tqdm.auto import tqdm

# Use PyTorch 2.0+ API
from torch.amp import autocast
from typing import List, Optional, Tuple, Union, Dict, Any

from .config import CReSOConfiguration
from .logging import get_logger
from .exceptions import raise_training_error

from .model import CReSOModel
from .wave_model import CReSOWaveModel
from .utils import Standardizer, as_tensor, apply_wave_physics_initialization
from .validation import validate_positive_int

logger = get_logger(__name__)


def acf_peak_seeds(
    X_np: np.ndarray, input_dim: int, n_modes: int, max_lag: int = 12
) -> List[Tuple[int, float]]:
    """Extract frequency seeds from autocorrelation function peaks.

    Args:
        X_np: Input data (N, input_dim) as numpy array
        input_dim: Input dimension
        n_modes: Number of frequency modes to extract
        max_lag: Maximum lag for ACF computation

    Returns:
        List of (dimension, frequency) tuples
    """
    validate_positive_int(input_dim, "input_dim")
    validate_positive_int(n_modes, "n_modes")

    logger.debug(
        "Extracting ACF peak seeds",
        extra={
            "input_dim": input_dim,
            "n_modes": n_modes,
            "max_lag": max_lag,
            "data_shape": X_np.shape,
        },
    )
    seeds = []

    for dim in range(input_dim):
        signal = X_np[:, dim]

        # Compute autocorrelation
        if len(signal) < max_lag * 2:
            continue

        # Normalize signal
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

        # Compute ACF using FFT for O(N log N) efficiency
        n = len(signal)
        signal_fft = np.fft.fft(signal, n=2 * n)
        acf_full = np.fft.ifft(signal_fft * np.conj(signal_fft)).real
        acf = acf_full[: max_lag + 1]  # Take positive lags only
        # Normalize (avoid division by zero and numerical issues)
        if acf[0] > 1e-10:
            acf = acf / acf[0]
        else:
            logger.debug(f"ACF[0] is too small ({acf[0]:.2e}), skipping normalization")

        # Find peaks (simple approach: local maxima)
        peaks = []
        for i in range(1, len(acf) - 1):
            if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.1:
                peaks.append((i, acf[i]))

        # Sort by peak height and take top ones
        peaks.sort(key=lambda x: x[1], reverse=True)

        for lag, strength in peaks[: min(2, len(peaks))]:
            if lag > 0:
                # Convert lag to frequency (assuming unit sampling)
                freq = 2 * np.pi / lag
                seeds.append((dim, freq))

    # Return top n_modes by strength (stored in peak tuples)
    return seeds[:n_modes]


def apply_freq_seeds(
    model: CReSOModel, X_train: torch.Tensor, frac: float = 0.2
) -> None:
    """Apply ACF-derived frequency seeds to model.

    Args:
        model: CReSO model to seed
        X_train: Training data for ACF analysis
        frac: Fraction of frequencies to seed
    """
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)

    X_np = X_train.cpu().numpy()
    input_dim = X_train.size(1)
    n_components = model.wave_layer.n_components

    n_seeds = max(1, int(frac * n_components))
    seeds = acf_peak_seeds(X_np, input_dim, n_seeds)

    logger.info(
        "Applying frequency seeds",
        extra={
            "n_seeds": n_seeds,
            "total_components": n_components,
            "seed_fraction": frac,
        },
    )

    if not seeds:
        return

    with torch.no_grad():
        for i, (dim, freq) in enumerate(seeds[:n_seeds]):
            if i < n_components:
                # Set frequency for this mode
                model.wave_layer.omega.data[i, dim] = freq
                # Zero out other dimensions for this mode to make it dimension-specific
                model.wave_layer.omega.data[i, :dim] = 0
                model.wave_layer.omega.data[i, dim + 1 :] = 0


class CReSOTrainer:
    """Training utilities for CReSO models.

    Provides training loop with early stopping, AMP, frequency seeding,
    and comprehensive regularization.
    """

    def __init__(self, config: CReSOConfiguration):
        self.config = config

        logger.info(
            "Initializing CReSOTrainer",
            extra={
                "config_type": type(config).__name__,
                "device": config.system.device,
            },
        )

    def fit(
        self,
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        class_weight: Optional[Union[str, Dict, float]] = None,
        standardize: bool = True,
        max_epochs: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_frequency: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        checkpoint_interval: Optional[int] = None,
        save_best_only: bool = False,
        verbose: int = 0,
    ) -> Tuple[
        Union[CReSOModel, CReSOWaveModel],
        torch.optim.Optimizer,
        Optional[Standardizer],
        Dict,
    ]:
        """Fit CReSO model to training data.

        Args:
            X_train: Training features (N, input_dim)
            y_train: Training labels (N,) - binary classification
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            class_weight: Class weighting ('balanced', dict, or positive class weight)
            standardize: Whether to standardize features
            max_epochs: Override config epochs if provided
            checkpoint_path: Path to save checkpoints
            checkpoint_frequency: How often to save checkpoints
            checkpoint_dir: Directory for checkpoints
            resume_from_checkpoint: Path to checkpoint to resume from
            checkpoint_interval: Alias for checkpoint_frequency
            verbose: Verbosity level

        Returns:
            Tuple of (trained model, optimizer, standardizer, history dict)
        """
        # Handle checkpoint parameters
        if checkpoint_interval is not None and checkpoint_frequency is None:
            checkpoint_frequency = checkpoint_interval

        # Handle epoch override
        original_epochs = self.config.training.max_epochs
        if max_epochs is not None:
            epochs = max_epochs
        else:
            epochs = original_epochs

        # Initialize history tracking
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }

        # Get device from config
        device = torch.device(self.config.system.device)

        if verbose > 0:
            logger.info(f"Starting training on device: {device}")

        # Convert to tensors
        X_train = as_tensor(X_train, device=device)
        y_train = as_tensor(y_train, device=device).float()

        if y_train.dim() > 1:
            y_train = y_train.squeeze()

        # Validation data
        has_validation = X_val is not None and y_val is not None
        if has_validation:
            X_val = as_tensor(X_val, device=device)
            y_val = as_tensor(y_val, device=device).float()
            if y_val.dim() > 1:
                y_val = y_val.squeeze()

        # Standardization
        standardizer = None
        if standardize:
            standardizer = Standardizer().to(device)
            X_train = standardizer.fit_transform(X_train)
            if has_validation:
                X_val = standardizer.transform(X_val)

        # Create model (wave physics or standard)
        if self.config.wave_physics.enable_wave_physics:
            model = CReSOWaveModel(
                self.config,
                use_wave_physics=True,
                n_propagation_steps=self.config.wave_physics.n_propagation_steps,
            ).to(device)
            logger.info("Using CReSOWaveModel with wave physics")
        else:
            model = CReSOModel(self.config).to(device)
            logger.info("Using standard CReSOModel")

        # Apply initialization
        if isinstance(model, CReSOWaveModel):
            # Apply wave physics initialization
            apply_wave_physics_initialization(
                model,
                X_train,
                q_factor=self.config.wave_physics.q_factor,
                phase_diversity=1.0,
                use_kmeans_centers=True,
            )
        else:
            # Apply standard frequency seeding
            use_seeds = self.config.frequency_seeding.enable_frequency_seeding
            seed_frac = self.config.frequency_seeding.seeding_fraction

            if use_seeds:
                apply_freq_seeds(model, X_train, seed_frac)

        # Compute class weights
        pos_weight = None
        if class_weight is not None:
            if class_weight == "balanced":
                n_pos = torch.sum(y_train).item()
                n_neg = len(y_train) - n_pos

                # Handle edge cases for single-class datasets
                if n_pos == 0:
                    logger.warning(
                        "No positive samples found, disabling class weighting"
                    )
                    pos_weight = None  # Don't apply any weighting
                elif n_neg == 0:
                    logger.warning(
                        "No negative samples found, disabling class weighting"
                    )
                    pos_weight = None  # Don't apply any weighting
                else:
                    pos_weight = torch.tensor(n_neg / n_pos, device=device)
            elif isinstance(class_weight, (int, float)):
                pos_weight = torch.tensor(float(class_weight), device=device)
            elif isinstance(class_weight, dict):
                pos_weight = torch.tensor(
                    class_weight.get(1, 1.0) / class_weight.get(0, 1.0), device=device
                )

        # Setup training
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Get training parameters
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay
        use_amp = self.config.training.use_automatic_mixed_precision
        grad_clip_norm = self.config.training.gradient_clip_norm
        batch_size = self.config.training.batch_size

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = self._create_scheduler(optimizer, epochs)

        # AMP setup
        from torch.amp import GradScaler

        scaler = GradScaler("cuda") if use_amp else None

        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            checkpoint = self.load_checkpoint(resume_from_checkpoint)

            # Load model state
            model.load_state_dict(checkpoint["model_state_dict"])

            # Load optimizer state
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load scheduler state if available
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # Resume from next epoch
            start_epoch = checkpoint.get("epoch", 0) + 1

            if "history" in checkpoint:
                # Restore previous training history
                for key in history.keys():
                    if key in checkpoint["history"]:
                        history[key] = checkpoint["history"][key]
            logger.info(f"Resumed training from epoch {start_epoch}")
        elif resume_from_checkpoint:
            logger.warning(f"Checkpoint file not found: {resume_from_checkpoint}")

        # Early stopping
        best_val_acc = -1
        best_model_state = None
        patience_counter = 0
        patience = 6

        # Training loop
        model.train()

        # Create progress bars for epochs and batches
        epoch_pbar = tqdm(
            range(start_epoch, epochs),
            desc="Training",
            unit="epoch",
            disable=verbose < 1,
        )

        for epoch in epoch_pbar:
            # Training phase
            train_loss = 0.0
            train_correct = 0
            n_batches = 0

            # Simple batch iteration with local permutation tensor
            n_samples = X_train.size(0)
            indices = torch.randperm(n_samples, device=device)

            # Create batch progress bar
            total_batches = (n_samples + batch_size - 1) // batch_size
            batch_pbar = tqdm(
                range(0, n_samples, batch_size),
                desc=f"Epoch {epoch+1}/{epochs}",
                leave=False,
                disable=verbose < 2,
                total=total_batches,
                unit="batch",
            )

            for start_idx in batch_pbar:
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                optimizer.zero_grad()

                # Forward pass with AMP
                if use_amp and scaler is not None:
                    with autocast("cuda"):
                        z, _, _, _, wave_info = model(X_batch, train_mode=True)
                        loss = criterion(z.squeeze(), y_batch)
                        loss = loss + model.regularization()

                        # Add wave physics regularization if applicable
                        if hasattr(model, "compute_wave_regularization_loss"):
                            wave_reg_loss = model.compute_wave_regularization_loss(
                                X_batch,
                                wave_info,
                                energy_weight=self.config.wave_physics.energy_conservation_weight,
                                pde_weight=self.config.wave_physics.pde_residual_weight,
                                q_constraint_weight=self.config.wave_physics.q_constraint_weight,
                            )
                            loss = loss + wave_reg_loss

                    scaler.scale(loss).backward()

                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), grad_clip_norm
                        )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    z, _, _, _, wave_info = model(X_batch, train_mode=True)
                    loss = criterion(z.squeeze(), y_batch)
                    loss = loss + model.regularization()

                    # Add wave physics regularization if applicable
                    if hasattr(model, "compute_wave_regularization_loss"):
                        wave_reg_loss = model.compute_wave_regularization_loss(
                            X_batch,
                            wave_info,
                            energy_weight=self.config.wave_physics.energy_conservation_weight,
                            pde_weight=self.config.wave_physics.pde_residual_weight,
                            q_constraint_weight=self.config.wave_physics.q_constraint_weight,
                        )
                        loss = loss + wave_reg_loss

                    loss.backward()

                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), grad_clip_norm
                        )

                    optimizer.step()

                # Track metrics
                with torch.no_grad():
                    # Check for training anomalies
                    loss_val = loss.item()
                    if not torch.isfinite(loss):
                        raise_training_error(
                            "Training loss became non-finite",
                            epoch=epoch,
                            batch=start_idx // batch_size,
                            loss_value=loss_val,
                        )

                    train_loss += loss_val
                    preds = torch.sigmoid(z.squeeze()) > 0.5
                    train_correct += torch.sum(preds == y_batch).item()
                    n_batches += 1

                    # Update batch progress bar with loss
                    batch_pbar.set_postfix(
                        {
                            "loss": f"{loss_val:.4f}",
                            "acc": f"{train_correct/(n_batches * batch_size):.4f}",
                        }
                    )

            # Validation phase
            val_acc = 0.0
            val_loss = None
            if has_validation:
                model.eval()
                with torch.no_grad():
                    z_val, _, _, _, _ = model(X_val, train_mode=False)
                    val_preds = torch.sigmoid(z_val.squeeze()) > 0.5
                    val_acc = torch.sum(val_preds == y_val).item() / len(y_val)
                    # Compute validation loss
                    val_loss = criterion(z_val.squeeze(), y_val).item()

            # Step scheduler (different logic for different scheduler types)
            if self.config.training.scheduler_type == "reduce_on_plateau":
                if has_validation:
                    scheduler.step(val_acc)  # Use validation accuracy
                else:
                    scheduler.step(train_loss / n_batches)  # Use training loss
            else:
                scheduler.step()

            # Return to training mode
            if has_validation:
                model.train()

            # Early stopping check
            if has_validation:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    epoch_pbar.set_description("Training (Early Stop)")
                    logger.info(
                        "Early stopping at epoch %d - best val_acc: %.4f",
                        epoch + 1,
                        best_val_acc,
                    )
                    break

            # Apply pruning if enabled (after some warmup epochs)
            if (
                self.config.wave_physics.enable_pruning
                and hasattr(model, "prune_wave_components")
                and epoch > 0.3 * epochs  # After 30% of training
                and epoch % max(1, epochs // 10) == 0
            ):  # Every 10% of epochs

                pruning_stats = model.prune_wave_components(
                    self.config.wave_physics.pruning_threshold
                )
                logger.info(
                    "Pruned %d/%d components (%.1f%% sparsity) at epoch %d",
                    pruning_stats["n_pruned"],
                    pruning_stats["total_components"],
                    pruning_stats["sparsity"] * 100,
                    epoch + 1,
                )

            # Progress update
            train_acc = train_correct / n_samples if n_samples > 0 else 0.0
            epoch_train_loss = train_loss / n_batches if n_batches > 0 else 0.0

            if has_validation:
                if verbose > 0:
                    logger.info(
                        "Epoch %d/%d: Loss=%.4f, Train Acc=%.4f, Val Acc=%.4f",
                        epoch + 1,
                        epochs,
                        epoch_train_loss,
                        train_acc,
                        val_acc,
                    )
            else:
                if verbose > 0:
                    logger.info(
                        "Epoch %d/%d: Loss=%.4f, Train Acc=%.4f",
                        epoch + 1,
                        epochs,
                        epoch_train_loss,
                        train_acc,
                    )

            # Track history
            history["train_loss"].append(epoch_train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss if has_validation else None)
            history["val_acc"].append(val_acc if has_validation else None)
            history["learning_rate"].append(optimizer.param_groups[0]["lr"])

            # Update epoch progress bar with metrics
            postfix = {
                "train_loss": f"{epoch_train_loss:.4f}",
                "train_acc": f"{train_acc:.4f}",
            }
            if has_validation:
                postfix.update(
                    {"val_loss": f"{val_loss:.4f}", "val_acc": f"{val_acc:.4f}"}
                )
            epoch_pbar.set_postfix(postfix)

            # Checkpoint saving
            should_save_checkpoint = False
            if checkpoint_frequency and (epoch + 1) % checkpoint_frequency == 0:
                if save_best_only and has_validation:
                    # Only save if validation accuracy improved
                    should_save_checkpoint = val_acc > best_val_acc
                elif not save_best_only:
                    # Save on frequency regardless of performance
                    should_save_checkpoint = True

                if should_save_checkpoint:
                    if checkpoint_path:
                        # Add best checkpoint metadata if this is a best-only save
                        extra_metadata = {}
                        if save_best_only and has_validation:
                            extra_metadata["best_val_loss"] = val_loss
                            extra_metadata["best_val_accuracy"] = val_acc
                            extra_metadata["is_best_checkpoint"] = True

                        self.save_checkpoint(
                            checkpoint_path,
                            model,
                            optimizer,
                            epoch,
                            epoch_train_loss,
                            val_loss=val_loss,
                            val_accuracy=val_acc if has_validation else None,
                            standardizer=standardizer,
                            scheduler=scheduler,
                            history=history,
                            **extra_metadata,
                        )
                    elif checkpoint_dir:
                        checkpoint_file = f"checkpoint_epoch_{epoch+1}.pt"
                        checkpoint_full_path = os.path.join(
                            checkpoint_dir, checkpoint_file
                        )
                        self.save_checkpoint(
                            checkpoint_full_path,
                            model,
                            optimizer,
                            epoch,
                            epoch_train_loss,
                            val_loss=val_loss,
                            val_accuracy=val_acc if has_validation else None,
                            standardizer=standardizer,
                            scheduler=scheduler,
                            history=history,
                        )

        # Restore best model if validation was used
        if has_validation and best_model_state is not None:
            model.load_state_dict(best_model_state)
            # Clean up best model state to free memory
            del best_model_state

        # Save final checkpoint if checkpoint_path is provided
        if checkpoint_path:
            # Add best checkpoint metadata if this is a best-only save
            extra_metadata = {}
            if save_best_only and has_validation:
                extra_metadata["best_val_loss"] = (
                    history["val_loss"][-1] if history["val_loss"] else None
                )
                extra_metadata["best_val_accuracy"] = (
                    history["val_acc"][-1] if history["val_acc"] else None
                )
                extra_metadata["is_best_checkpoint"] = True

            self.save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epochs - 1,  # Last completed epoch
                history["train_loss"][-1] if history["train_loss"] else 0.0,
                val_loss=history["val_loss"][-1] if history["val_loss"] else None,
                val_accuracy=history["val_acc"][-1] if history["val_acc"] else None,
                standardizer=standardizer,
                scheduler=scheduler,
                history=history,
                **extra_metadata,
            )

        model.eval()
        return model, optimizer, standardizer, history

    def predict_proba(
        self,
        model: CReSOModel,
        X: Union[np.ndarray, torch.Tensor],
        standardizer: Optional[Standardizer] = None,
    ) -> np.ndarray:
        """Predict class probabilities.

        Args:
            model: Trained model
            X: Input features
            standardizer: Fitted standardizer (optional)

        Returns:
            Predicted probabilities (N, 2) for [neg_class, pos_class]
        """
        device = next(model.parameters()).device
        X = as_tensor(X, device=device)

        if standardizer is not None:
            X = standardizer.transform(X)

        model.eval()
        with torch.no_grad():
            z, _, _, _, _ = model(X, train_mode=False)
            pos_probs = torch.sigmoid(z.squeeze()).cpu().numpy()

        # Return probabilities for both classes
        neg_probs = 1 - pos_probs
        return np.column_stack([neg_probs, pos_probs])

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        model: Union[CReSOModel, CReSOWaveModel],
        optimizer: optim.Optimizer,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        standardizer: Optional[Standardizer] = None,
        scheduler: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """Save training checkpoint.

        Args:
            filepath: Path to save checkpoint
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss (optional)
            val_accuracy: Validation accuracy (optional)
            standardizer: Data standardizer (optional)
            scheduler: Learning rate scheduler (optional)
            **kwargs: Additional metadata to save
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "config": self.config.to_dict(),
            "model_class": type(model).__name__,
        }

        if standardizer is not None:
            checkpoint["standardizer_state_dict"] = standardizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Add any additional metadata
        checkpoint.update(kwargs)

        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint at epoch {epoch} to {filepath}")

    def load_checkpoint(
        self, filepath: Union[str, Path], device: Optional[torch.device] = None
    ) -> Dict:
        """Load training checkpoint.

        Args:
            filepath: Path to checkpoint file
            device: Device to load checkpoint to

        Returns:
            Dictionary containing checkpoint data
        """
        if device is None:
            device = torch.device(self.config.system.device)

        from .config import (
            CReSOConfiguration,
            ModelArchitectureConfig,
            TrainingConfig,
            RegularizationConfig,
            FrequencySeedingConfig,
            SystemConfig,
            WavePhysicsConfig,
        )

        safe_classes = [
            CReSOConfiguration,
            ModelArchitectureConfig,
            TrainingConfig,
            RegularizationConfig,
            FrequencySeedingConfig,
            SystemConfig,
            WavePhysicsConfig,
        ]
        with torch.serialization.safe_globals(safe_classes):
            checkpoint = torch.load(filepath, map_location=device, weights_only=True)
        logger.info(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} from {filepath}"
        )

        return checkpoint

    def resume_training(
        self,
        checkpoint_path: Union[str, Path],
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        X_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        class_weight: Optional[Union[str, Dict, float]] = None,
        additional_epochs: int = 0,
    ) -> Tuple[Union[CReSOModel, CReSOWaveModel], Optional[Standardizer]]:
        """Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            class_weight: Class weighting (optional)
            additional_epochs: Additional epochs to train beyond checkpoint

        Returns:
            Trained model and standardizer
        """
        # Load checkpoint
        checkpoint = self.load_checkpoint(checkpoint_path)

        # Create model based on checkpoint
        device = torch.device(self.config.system.device)

        if checkpoint["model_class"] == "CReSOWaveModel":
            model = CReSOWaveModel(
                self.config,
                use_wave_physics=True,
                n_propagation_steps=self.config.wave_physics.n_propagation_steps,
            ).to(device)
        else:
            model = CReSOModel(self.config).to(device)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load standardizer if exists
        standardizer = None
        if "standardizer_state_dict" in checkpoint:
            standardizer = Standardizer()
            standardizer.load_state_dict(checkpoint["standardizer_state_dict"])

        # Update config for additional training
        if additional_epochs > 0:
            start_epoch = checkpoint["epoch"]
            logger.info(
                f"Resuming training from epoch {start_epoch} for {additional_epochs} more epochs"
            )

            # Modify training loop to continue from checkpoint epoch
            # This would require modifying the main training loop, which is complex
            # For now, we'll return the loaded model and let the user retrain if needed
            logger.warning(
                "Additional training from checkpoint not fully implemented yet"
            )

        logger.info(
            f"Successfully resumed from checkpoint at epoch {checkpoint['epoch']}"
        )
        return checkpoint

    def _create_scheduler(self, optimizer: optim.Optimizer, epochs: int) -> Optional[
        Union[
            optim.lr_scheduler.ReduceLROnPlateau,
            optim.lr_scheduler.CosineAnnealingLR,
            optim.lr_scheduler.StepLR,
        ]
    ]:
        """Create learning rate scheduler based on configuration.

        Args:
            optimizer: The optimizer to schedule
            epochs: Total number of training epochs

        Returns:
            Learning rate scheduler
        """
        scheduler_type = self.config.training.scheduler_type

        # Define no-op scheduler class once
        class NoScheduler:
            def step(self, *args, **kwargs):
                pass

            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                pass

        if scheduler_type == "none":
            return NoScheduler()

        elif scheduler_type == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",  # For accuracy (higher is better)
                factor=self.config.training.scheduler_factor,
                patience=self.config.training.scheduler_patience,
                min_lr=self.config.training.scheduler_min_lr,
                verbose=True,
            )

        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=self.config.training.scheduler_min_lr,
                last_epoch=-1,
            )

        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.training.scheduler_step_size,
                gamma=self.config.training.scheduler_factor,
                last_epoch=-1,
            )

        else:
            # Fall back to no scheduler for invalid types
            logger.warning(
                f"Unknown scheduler type '{scheduler_type}', falling back to no scheduler"
            )
            return NoScheduler()

    def predict(
        self,
        model: CReSOModel,
        X: Union[np.ndarray, torch.Tensor],
        standardizer: Optional[Standardizer] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict class labels.

        Args:
            model: Trained model
            X: Input features
            standardizer: Fitted standardizer (optional)
            threshold: Classification threshold

        Returns:
            Predicted labels (N,)
        """
        proba = self.predict_proba(model, X, standardizer)
        return (proba[:, 1] >= threshold).astype(int)
