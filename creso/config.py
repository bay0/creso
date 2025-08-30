"""
Enhanced configuration system for CReSO models.

Provides comprehensive configuration with validation, presets,
and enterprise-grade parameter management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, Literal
import torch
from pathlib import Path

from .constants import (
    DEFAULT_N_COMPONENTS,
    DEFAULT_LOCALIZED,
    DEFAULT_LEARN_CENTERS,
    DEFAULT_INIT_FREQUENCY_SCALE,
    DEFAULT_N_GEOMETRIC_HIDDEN,
    MAX_INPUT_DIMENSIONS,
    MAX_N_COMPONENTS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_FREQUENCY_DROPOUT,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_USE_AMP,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    MAX_BATCH_SIZE,
    DEFAULT_L2_FREQUENCY,
    DEFAULT_GROUP_L1,
    DEFAULT_CENTER_DISPERSION,
    DEFAULT_USE_SEED_FREQUENCIES,
    DEFAULT_SEED_FRACTION,
    DEFAULT_MAX_ACF_LAG,
    DEFAULT_DEVICE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PIN_MEMORY,
    CONFIGURATION_PRESETS,
)
from .validation import (
    validate_positive_int,
    validate_positive_float,
    validate_probability,
)
from .exceptions import raise_configuration_error
from .logging import get_logger

logger = get_logger(__name__)

# Type aliases for better readability
Device = Union[str, torch.device]
PresetName = Literal["fast", "balanced", "accurate", "minimal"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


@dataclass
class ModelArchitectureConfig:
    """Configuration for model architecture parameters.

    Args:
        input_dim: Number of input features
        n_components: Number of spectral components (formerly K)
        use_localized_envelopes: Whether to use Gaussian envelope localization
        learn_envelope_centers: Whether to learn envelope centers
        initial_frequency_scale: Scale for random frequency initialization
        geometric_hidden_dim: Hidden units in geometric processing path
    """

    input_dim: int
    n_components: int = DEFAULT_N_COMPONENTS
    use_localized_envelopes: bool = DEFAULT_LOCALIZED
    learn_envelope_centers: bool = DEFAULT_LEARN_CENTERS
    initial_frequency_scale: float = DEFAULT_INIT_FREQUENCY_SCALE
    geometric_hidden_dim: int = DEFAULT_N_GEOMETRIC_HIDDEN

    def __post_init__(self):
        """Validate architecture configuration."""
        self.input_dim = validate_positive_int(
            self.input_dim, "input_dim", minimum=1, maximum=MAX_INPUT_DIMENSIONS
        )
        self.n_components = validate_positive_int(
            self.n_components, "n_components", minimum=1, maximum=MAX_N_COMPONENTS
        )
        self.initial_frequency_scale = validate_positive_float(
            self.initial_frequency_scale,
            "initial_frequency_scale",
            minimum=0.1,
            maximum=10.0,
        )
        self.geometric_hidden_dim = validate_positive_int(
            self.geometric_hidden_dim, "geometric_hidden_dim", minimum=1, maximum=1000
        )


@dataclass
class TrainingConfig:
    """Configuration for training parameters.

    Args:
        learning_rate: Optimizer learning rate
        max_epochs: Maximum number of training epochs
        batch_size: Training batch size
        weight_decay: L2 regularization strength
        frequency_dropout_probability: Probability of dropping spectral components during training
        gradient_clip_norm: Maximum gradient norm (None to disable clipping)
        use_automatic_mixed_precision: Whether to use automatic mixed precision
        early_stopping_patience: Epochs to wait before early stopping
        scheduler_type: Learning rate scheduler type ('none', 'reduce_on_plateau', 'cosine', 'step')
        scheduler_patience: Patience for ReduceLROnPlateau scheduler
        scheduler_factor: Factor to reduce learning rate by
        scheduler_step_size: Step size for StepLR scheduler
        scheduler_min_lr: Minimum learning rate for schedulers
    """

    learning_rate: float = DEFAULT_LEARNING_RATE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    frequency_dropout_probability: float = DEFAULT_FREQUENCY_DROPOUT
    gradient_clip_norm: Optional[float] = DEFAULT_GRADIENT_CLIP_NORM
    use_automatic_mixed_precision: bool = DEFAULT_USE_AMP
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    scheduler_type: str = "none"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_step_size: int = 30
    scheduler_min_lr: float = 1e-6

    def __post_init__(self):
        """Validate training configuration."""
        self.learning_rate = validate_positive_float(
            self.learning_rate, "learning_rate", minimum=1e-6, maximum=1.0
        )
        self.max_epochs = validate_positive_int(
            self.max_epochs, "max_epochs", minimum=1, maximum=MAX_EPOCHS
        )
        self.batch_size = validate_positive_int(
            self.batch_size, "batch_size", minimum=1, maximum=MAX_BATCH_SIZE
        )
        self.weight_decay = validate_positive_float(
            self.weight_decay, "weight_decay", minimum=0.0, maximum=1.0
        )
        self.frequency_dropout_probability = validate_probability(
            self.frequency_dropout_probability, "frequency_dropout_probability"
        )
        if self.gradient_clip_norm is not None:
            self.gradient_clip_norm = validate_positive_float(
                self.gradient_clip_norm,
                "gradient_clip_norm",
                minimum=0.1,
                maximum=100.0,
            )
        self.early_stopping_patience = validate_positive_int(
            self.early_stopping_patience,
            "early_stopping_patience",
            minimum=1,
            maximum=100,
        )

        # Validate scheduler parameters
        valid_schedulers = ["none", "reduce_on_plateau", "cosine", "step"]
        if self.scheduler_type not in valid_schedulers:
            raise ValueError(
                f"scheduler_type must be one of {valid_schedulers}, got {self.scheduler_type}"
            )

        self.scheduler_patience = validate_positive_int(
            self.scheduler_patience, "scheduler_patience", minimum=1, maximum=100
        )
        self.scheduler_factor = validate_positive_float(
            self.scheduler_factor, "scheduler_factor", minimum=0.01, maximum=0.99
        )
        self.scheduler_step_size = validate_positive_int(
            self.scheduler_step_size, "scheduler_step_size", minimum=1, maximum=1000
        )
        self.scheduler_min_lr = validate_positive_float(
            self.scheduler_min_lr, "scheduler_min_lr", minimum=1e-8, maximum=1e-2
        )


@dataclass
class RegularizationConfig:
    """Configuration for regularization parameters.

    Args:
        l2_frequency_penalty: L2 regularization on frequency parameters
        group_l1_amplitude_penalty: Group L1 regularization on amplitude parameters
        center_dispersion_penalty: Penalty encouraging diverse envelope centers
    """

    l2_frequency_penalty: float = DEFAULT_L2_FREQUENCY
    group_l1_amplitude_penalty: float = DEFAULT_GROUP_L1
    center_dispersion_penalty: float = DEFAULT_CENTER_DISPERSION

    def __post_init__(self):
        """Validate regularization configuration."""
        self.l2_frequency_penalty = validate_positive_float(
            self.l2_frequency_penalty, "l2_frequency_penalty", minimum=0.0, maximum=1.0
        )
        self.group_l1_amplitude_penalty = validate_positive_float(
            self.group_l1_amplitude_penalty,
            "group_l1_amplitude_penalty",
            minimum=0.0,
            maximum=1.0,
        )
        self.center_dispersion_penalty = validate_positive_float(
            self.center_dispersion_penalty,
            "center_dispersion_penalty",
            minimum=0.0,
            maximum=1.0,
        )


@dataclass
class FrequencySeedingConfig:
    """Configuration for ACF-based frequency seeding.

    Args:
        enable_frequency_seeding: Whether to initialize frequencies using ACF analysis
        seeding_fraction: Fraction of components to initialize with ACF seeds
        max_autocorr_lag: Maximum lag for autocorrelation analysis
    """

    enable_frequency_seeding: bool = DEFAULT_USE_SEED_FREQUENCIES
    seeding_fraction: float = DEFAULT_SEED_FRACTION
    max_autocorr_lag: int = DEFAULT_MAX_ACF_LAG

    def __post_init__(self):
        """Validate frequency seeding configuration."""
        self.seeding_fraction = validate_probability(
            self.seeding_fraction, "seeding_fraction"
        )
        self.max_autocorr_lag = validate_positive_int(
            self.max_autocorr_lag, "max_autocorr_lag", minimum=2, maximum=1000
        )


@dataclass
class SystemConfig:
    """Configuration for system-level parameters.

    Args:
        device: PyTorch device specification
        random_seed: Random seed for reproducibility
        num_workers: Number of worker threads for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
    """

    device: str = DEFAULT_DEVICE
    random_seed: int = 42
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_memory: bool = DEFAULT_PIN_MEMORY

    def __post_init__(self):
        """Validate system configuration."""
        self.random_seed = validate_positive_int(
            self.random_seed, "random_seed", minimum=0, maximum=2**31 - 1
        )
        self.num_workers = validate_positive_int(
            self.num_workers, "num_workers", minimum=0, maximum=32
        )

        # Validate device and fallback if necessary
        self._validate_and_setup_device()

    def _validate_and_setup_device(self):
        """Validate device availability and setup fallback."""
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"


@dataclass
class WavePhysicsConfig:
    """Configuration for wave physics components.

    Args:
        enable_wave_physics: Whether to use wave physics components
        q_factor: Constant Q-factor (bandwidth/frequency ratio)
        n_propagation_steps: Number of wave propagation layers
        use_coherence_gate: Enable interference-based coherence gating
        use_dispersive_packets: Use dispersive wave packets with learnable wave speeds
        energy_conservation_weight: Weight for energy conservation regularization
        pde_residual_weight: Weight for PDE residual regularization
        q_constraint_weight: Weight for constant-Q constraint regularization
        dispersion_hidden_dim: Hidden dimension for dispersion relation network
        coherence_temperature: Temperature parameter for coherence gating
        enable_pruning: Enable component pruning during training
        pruning_threshold: Percentile threshold for pruning (0-100)
        streaming_window_size: Buffer size for streaming mode (if used)
    """

    enable_wave_physics: bool = False
    q_factor: float = 1.0
    n_propagation_steps: int = 2
    use_coherence_gate: bool = True
    use_dispersive_packets: bool = False
    energy_conservation_weight: float = 0.1
    pde_residual_weight: float = 0.01
    q_constraint_weight: float = 0.01
    dispersion_hidden_dim: int = 16
    coherence_temperature: float = 1.0
    enable_pruning: bool = False
    pruning_threshold: float = 50.0
    streaming_window_size: int = 32

    def __post_init__(self):
        """Validate wave physics configuration."""
        self.q_factor = validate_positive_float(
            self.q_factor, "q_factor", minimum=0.1, maximum=10.0
        )
        self.n_propagation_steps = validate_positive_int(
            self.n_propagation_steps, "n_propagation_steps", minimum=1, maximum=10
        )
        self.energy_conservation_weight = validate_positive_float(
            self.energy_conservation_weight,
            "energy_conservation_weight",
            minimum=0.0,
            maximum=1.0,
        )
        self.pde_residual_weight = validate_positive_float(
            self.pde_residual_weight, "pde_residual_weight", minimum=0.0, maximum=1.0
        )
        self.q_constraint_weight = validate_positive_float(
            self.q_constraint_weight, "q_constraint_weight", minimum=0.0, maximum=1.0
        )
        self.dispersion_hidden_dim = validate_positive_int(
            self.dispersion_hidden_dim, "dispersion_hidden_dim", minimum=4, maximum=128
        )
        self.coherence_temperature = validate_positive_float(
            self.coherence_temperature,
            "coherence_temperature",
            minimum=0.1,
            maximum=10.0,
        )
        self.pruning_threshold = (
            validate_probability(self.pruning_threshold / 100.0, "pruning_threshold")
            * 100.0
        )  # Convert back to percentile
        self.streaming_window_size = validate_positive_int(
            self.streaming_window_size, "streaming_window_size", minimum=8, maximum=512
        )


@dataclass
class CReSOConfiguration:
    """Enhanced configuration for CReSO models.

    Provides comprehensive configuration with validation, type safety,
    and enterprise-grade parameter management.

    Args:
        architecture: Model architecture configuration
        training: Training configuration
        regularization: Regularization configuration
        frequency_seeding: Frequency seeding configuration
        wave_physics: Wave physics configuration
        system: System configuration
        name: Configuration name for identification
        description: Human-readable description
    """

    architecture: ModelArchitectureConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    frequency_seeding: FrequencySeedingConfig = field(
        default_factory=FrequencySeedingConfig
    )
    wave_physics: WavePhysicsConfig = field(default_factory=WavePhysicsConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    name: str = "default"
    description: str = "Default CReSO configuration"

    def __post_init__(self):
        """Validate complete configuration and check compatibility."""
        self._validate_configuration_compatibility()
        logger.info(
            f"Initialized CReSO configuration '{self.name}'",
            extra={
                "input_dim": self.architecture.input_dim,
                "n_components": self.architecture.n_components,
                "max_epochs": self.training.max_epochs,
            },
        )

    def _validate_configuration_compatibility(self):
        """Check for incompatible parameter combinations."""
        # Check if frequency seeding is compatible with architecture
        if (
            self.frequency_seeding.enable_frequency_seeding
            and not self.architecture.use_localized_envelopes
        ):
            logger.warning("Frequency seeding works best with localized envelopes")

        # Check batch size vs model complexity
        model_complexity = self.architecture.n_components * self.architecture.input_dim
        if model_complexity > 10000 and self.training.batch_size < 64:
            logger.warning(
                "Small batch size with complex model may cause training instability"
            )

        # Check AMP compatibility
        if self.training.use_automatic_mixed_precision and self.system.device == "cpu":
            logger.warning("AMP requested but using CPU device, disabling AMP")
            self.training.use_automatic_mixed_precision = False

    @classmethod
    def from_preset(
        cls, preset_name: PresetName, input_dim: int, **overrides: Any
    ) -> CReSOConfiguration:
        """Create configuration from preset.

        Args:
            preset_name: Name of preset configuration
            input_dim: Input dimension for the model
            **overrides: Parameter overrides

        Returns:
            Configuration instance
        """
        if preset_name not in CONFIGURATION_PRESETS:
            raise_configuration_error(
                f"Unknown preset: {preset_name}",
                parameter="preset_name",
                expected=list(CONFIGURATION_PRESETS.keys()),
            )

        preset = CONFIGURATION_PRESETS[preset_name].copy()
        preset.update(overrides)

        # Create architecture config
        architecture = ModelArchitectureConfig(
            input_dim=input_dim,
            n_components=preset.get("n_components", DEFAULT_N_COMPONENTS),
        )

        # Create training config
        training = TrainingConfig(
            learning_rate=preset.get("learning_rate", DEFAULT_LEARNING_RATE),
            max_epochs=preset.get("max_epochs", DEFAULT_MAX_EPOCHS),
            batch_size=preset.get("batch_size", DEFAULT_BATCH_SIZE),
            use_automatic_mixed_precision=preset.get("use_amp", DEFAULT_USE_AMP),
            frequency_dropout_probability=preset.get(
                "frequency_dropout", DEFAULT_FREQUENCY_DROPOUT
            ),
        )

        return cls(
            architecture=architecture,
            training=training,
            name=f"{preset_name}_preset",
            description=f"Preset configuration for {preset_name} performance",
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> CReSOConfiguration:
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Configuration instance
        """
        # Extract nested configurations
        arch_config = ModelArchitectureConfig(**config_dict.pop("architecture", {}))
        train_config = TrainingConfig(**config_dict.pop("training", {}))
        reg_config = RegularizationConfig(**config_dict.pop("regularization", {}))
        freq_config = FrequencySeedingConfig(**config_dict.pop("frequency_seeding", {}))
        sys_config = SystemConfig(**config_dict.pop("system", {}))

        return cls(
            architecture=arch_config,
            training=train_config,
            regularization=reg_config,
            frequency_seeding=freq_config,
            system=sys_config,
            **config_dict,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as nested dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "architecture": {
                "input_dim": self.architecture.input_dim,
                "n_components": self.architecture.n_components,
                "use_localized_envelopes": self.architecture.use_localized_envelopes,
                "learn_envelope_centers": self.architecture.learn_envelope_centers,
                "initial_frequency_scale": self.architecture.initial_frequency_scale,
                "geometric_hidden_dim": self.architecture.geometric_hidden_dim,
            },
            "training": {
                "learning_rate": self.training.learning_rate,
                "max_epochs": self.training.max_epochs,
                "batch_size": self.training.batch_size,
                "weight_decay": self.training.weight_decay,
                "frequency_dropout_probability": self.training.frequency_dropout_probability,
                "gradient_clip_norm": self.training.gradient_clip_norm,
                "use_automatic_mixed_precision": self.training.use_automatic_mixed_precision,
                "early_stopping_patience": self.training.early_stopping_patience,
            },
            "regularization": {
                "l2_frequency_penalty": self.regularization.l2_frequency_penalty,
                "group_l1_amplitude_penalty": self.regularization.group_l1_amplitude_penalty,
                "center_dispersion_penalty": self.regularization.center_dispersion_penalty,
            },
            "frequency_seeding": {
                "enable_frequency_seeding": self.frequency_seeding.enable_frequency_seeding,
                "seeding_fraction": self.frequency_seeding.seeding_fraction,
                "max_autocorr_lag": self.frequency_seeding.max_autocorr_lag,
            },
            "wave_physics": {
                "enable_wave_physics": self.wave_physics.enable_wave_physics,
                "q_factor": self.wave_physics.q_factor,
                "n_propagation_steps": self.wave_physics.n_propagation_steps,
                "use_coherence_gate": self.wave_physics.use_coherence_gate,
                "energy_conservation_weight": self.wave_physics.energy_conservation_weight,
                "pde_residual_weight": self.wave_physics.pde_residual_weight,
                "q_constraint_weight": self.wave_physics.q_constraint_weight,
                "dispersion_hidden_dim": self.wave_physics.dispersion_hidden_dim,
                "coherence_temperature": self.wave_physics.coherence_temperature,
            },
            "system": {
                "device": self.system.device,
                "random_seed": self.system.random_seed,
                "num_workers": self.system.num_workers,
                "pin_memory": self.system.pin_memory,
            },
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to file.

        Args:
            filepath: Path to save configuration
        """
        import json

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved configuration to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> CReSOConfiguration:
        """Load configuration from file.

        Args:
            filepath: Path to configuration file

        Returns:
            Loaded configuration
        """
        import json

        filepath = Path(filepath)
        if not filepath.exists():
            raise_configuration_error(f"Configuration file not found: {filepath}")

        with open(filepath, "r") as f:
            config_dict = json.load(f)

        logger.info(f"Loaded configuration from {filepath}")
        return cls.from_dict(config_dict)


# Main configuration class
CReSOConfig = CReSOConfiguration  # Keep this for now during transition
