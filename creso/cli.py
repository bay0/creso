"""
Command-line interface for CReSO using Hydra configuration.
"""

import os
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore

from .config import CReSOConfiguration, ModelArchitectureConfig, TrainingConfig
from .classifier import CReSOClassifier, CReSOvRClassifier
from .utils import set_global_seed
from .logging import get_logger

logger = get_logger(__name__)


# Default configurations
@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    """Main CLI entrypoint."""

    # Convert OmegaConf to CReSOConfiguration
    model_cfg = OmegaConf.to_object(cfg.model)

    # Create architecture config
    arch_config = ModelArchitectureConfig(
        input_dim=model_cfg["input_dim"],
        n_components=model_cfg.get("n_components", 128),
        use_localized_envelopes=model_cfg.get("localized", True),
        learn_envelope_centers=model_cfg.get("learn_centers", True),
        initial_frequency_scale=model_cfg.get("init_freq_scale", 3.0),
        geometric_hidden_dim=model_cfg.get("geom_hidden", 64),
    )

    # Create training config
    train_config = TrainingConfig(
        learning_rate=model_cfg.get("learning_rate", 0.003),
        max_epochs=model_cfg.get("max_epochs", 25),
        batch_size=model_cfg.get("batch_size", 256),
        weight_decay=model_cfg.get("weight_decay", 0.0),
        frequency_dropout_probability=model_cfg.get("freq_dropout_p", 0.2),
        gradient_clip_norm=model_cfg.get("grad_clip_norm", 1.0),
        use_automatic_mixed_precision=model_cfg.get("use_amp", True),
    )

    # Create full configuration
    from .config import RegularizationConfig, FrequencySeedingConfig, SystemConfig

    reg_config = RegularizationConfig(
        l2_frequency_penalty=model_cfg.get("l2_freq", 1e-4),
        group_l1_amplitude_penalty=model_cfg.get("group_l1", 1e-3),
        center_dispersion_penalty=model_cfg.get("center_disp", 1e-5),
    )

    freq_config = FrequencySeedingConfig(
        enable_frequency_seeding=model_cfg.get("use_seed_freqs", True),
        seeding_fraction=model_cfg.get("seed_frac", 0.2),
    )

    sys_config = SystemConfig(
        device=model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        random_seed=model_cfg.get("random_seed", 42),
    )

    creso_config = CReSOConfiguration(
        architecture=arch_config,
        training=train_config,
        regularization=reg_config,
        frequency_seeding=freq_config,
        system=sys_config,
    )

    # Set global seed
    set_global_seed(creso_config.system.random_seed)

    # Create output directory
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Dispatch to task-specific function
    task = cfg.task

    if task == "tabular_binary":
        run_tabular_binary(cfg, creso_config)
    elif task == "tabular_multiclass":
        run_tabular_multiclass(cfg, creso_config)
    elif task == "timeseries_binary":
        run_timeseries_binary(cfg, creso_config)
    elif task == "graph_nodes_binary":
        run_graph_nodes_binary(cfg, creso_config)
    else:
        raise ValueError(f"Unknown task: {task}")


def run_tabular_binary(cfg: DictConfig, creso_config: CReSOConfiguration) -> None:
    """Run binary tabular classification."""
    logger.info("Running tabular binary classification...")

    # Load data
    data = np.load(cfg.data.path)
    X_train = data["Xtr"]
    y_train = data["ytr"]
    X_test = data["Xte"]
    y_test = data["yte"]

    logger.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

    # Update config with input dimension
    creso_config.architecture.input_dim = X_train.shape[1]

    # Train classifier
    clf = CReSOClassifier(creso_config)

    # Use class weighting if specified
    class_weight = cfg.data.get("class_weight", None)
    clf.fit(X_train, y_train, class_weight=class_weight)

    # Evaluate
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")

    # Save model
    model_path = os.path.join(cfg.out_dir, "model.pkl")
    clf.save(model_path)
    logger.info(f"Model saved to: {model_path}")

    # Save predictions
    test_proba = clf.predict_proba(X_test)
    test_preds = clf.predict(X_test)

    np.savez(
        os.path.join(cfg.out_dir, "predictions.npz"),
        test_proba=test_proba,
        test_preds=test_preds,
        test_true=y_test,
    )

    # Save metrics
    metrics = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_features": X_train.shape[1],
    }

    OmegaConf.save(metrics, os.path.join(cfg.out_dir, "metrics.yaml"))


def run_tabular_multiclass(cfg: DictConfig, creso_config: CReSOConfiguration) -> None:
    """Run multiclass tabular classification."""
    logger.info("Running tabular multiclass classification...")

    # Load data
    data = np.load(cfg.data.path)
    X_train = data["Xtr"]
    y_train = data["ytr"]
    X_test = data["Xte"]
    y_test = data["yte"]

    logger.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    logger.info(f"Number of classes: {len(np.unique(y_train))}")

    # Update config with input dimension
    creso_config.architecture.input_dim = X_train.shape[1]

    # Train one-vs-rest classifier
    clf = CReSOvRClassifier(creso_config)
    clf.fit(X_train, y_train)

    # Evaluate
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")

    # Save model
    model_dir = os.path.join(cfg.out_dir, "multiclass_model")
    clf.save(model_dir)
    logger.info(f"Model saved to: {model_dir}")

    # Save predictions
    test_proba = clf.predict_proba(X_test)
    test_preds = clf.predict(X_test)

    np.savez(
        os.path.join(cfg.out_dir, "predictions.npz"),
        test_proba=test_proba,
        test_preds=test_preds,
        test_true=y_test,
    )

    # Save metrics
    metrics = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_features": X_train.shape[1],
        "n_classes": len(np.unique(y_train)),
    }

    OmegaConf.save(metrics, os.path.join(cfg.out_dir, "metrics.yaml"))


def run_timeseries_binary(cfg: DictConfig, creso_config: CReSOConfiguration) -> None:
    """Run binary time-series classification."""
    try:
        from .adapters.timeseries import TimeSeriesCReSOClassifier
    except ImportError:
        raise ImportError("Time-series adapter not available")

    logger.info("Running time-series binary classification...")

    # Load data
    data = np.load(cfg.data.path)
    series_train = data["series_train"]
    target_train = data["target_train"]
    series_test = data["series_test"]
    target_test = data["target_test"]

    logger.info(f"Training series length: {len(series_train)}")
    logger.info(f"Test series length: {len(series_test)}")

    # Get time-series parameters
    ts_params = cfg.data.timeseries

    # Create time-series classifier
    clf = TimeSeriesCReSOClassifier(
        window=ts_params.window,
        horizon=ts_params.horizon,
        rates=ts_params.rates,
        step=ts_params.step,
        config=creso_config,
    )

    # Train
    clf.fit(series_train, target_train)

    # Evaluate
    train_acc = clf.score(series_train, target_train)
    test_acc = clf.score(series_test, target_test)

    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")

    # Save model
    model_path = os.path.join(cfg.out_dir, "ts_model.pkl")
    clf.save(model_path)
    logger.info(f"Model saved to: {model_path}")

    # Save predictions for test series
    test_proba = clf.predict_proba(series_test)
    test_preds = clf.predict(series_test)

    # Get corresponding true labels for the windows
    from .adapters.timeseries import make_multirate_windows

    _, y_test_windows = make_multirate_windows(
        series_test, target_test, clf.window, clf.horizon, clf.rates, clf.step
    )

    np.savez(
        os.path.join(cfg.out_dir, "predictions.npz"),
        test_proba=test_proba,
        test_preds=test_preds,
        test_true=y_test_windows,
    )

    # Save metrics
    metrics = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "train_series_length": len(series_train),
        "test_series_length": len(series_test),
        "window_size": ts_params.window,
        "horizon": ts_params.horizon,
        "rates": ts_params.rates,
    }

    OmegaConf.save(metrics, os.path.join(cfg.out_dir, "metrics.yaml"))


def run_graph_nodes_binary(cfg: DictConfig, creso_config: CReSOConfiguration) -> None:
    """Run binary graph node classification."""
    try:
        from .adapters.graph import GraphNodeCReSOClassifier
    except ImportError:
        raise ImportError("Graph adapter not available (requires scipy)")

    logger.info("Running graph node binary classification...")

    # Load data
    data = np.load(cfg.data.path)
    adj = data["adj"]
    X = data["X"]
    y = data["y"]
    train_mask = data.get("train_mask", None)
    test_mask = data.get("test_mask", None)

    logger.info(f"Graph: {adj.shape[0]} nodes, {X.shape[1]} features")

    if train_mask is not None:
        logger.info(f"Training nodes: {np.sum(train_mask)}")
    if test_mask is not None:
        logger.info(f"Test nodes: {np.sum(test_mask)}")

    # Get graph parameters
    graph_params = cfg.data.graph

    # Create graph classifier
    clf = GraphNodeCReSOClassifier(K=graph_params.K, config=creso_config)

    # Train
    val_mask = data.get("val_mask", None)
    clf.fit(adj, X, y, train_mask=train_mask, val_mask=val_mask)

    # Evaluate
    if train_mask is not None:
        train_acc = clf.score(adj, X, y, mask=train_mask)
        logger.info(f"Training accuracy: {train_acc:.4f}")

    if test_mask is not None:
        test_acc = clf.score(adj, X, y, mask=test_mask)
        logger.info(f"Test accuracy: {test_acc:.4f}")
    else:
        # Evaluate on all nodes if no test mask
        test_acc = clf.score(adj, X, y)
        logger.info(f"Overall accuracy: {test_acc:.4f}")

    # Save model
    model_path = os.path.join(cfg.out_dir, "graph_model.pkl")
    clf.save(model_path)
    logger.info(f"Model saved to: {model_path}")

    # Save predictions
    if test_mask is not None:
        test_proba = clf.predict_proba(adj, X, mask=test_mask)
        test_preds = clf.predict(adj, X, mask=test_mask)
        test_true = y[test_mask]
    else:
        test_proba = clf.predict_proba(adj, X)
        test_preds = clf.predict(adj, X)
        test_true = y

    np.savez(
        os.path.join(cfg.out_dir, "predictions.npz"),
        test_proba=test_proba,
        test_preds=test_preds,
        test_true=test_true,
    )

    # Save metrics
    metrics = {
        "test_accuracy": float(test_acc),
        "n_nodes": adj.shape[0],
        "n_features": X.shape[1],
        "n_edges": int(np.sum(adj > 0) // 2),  # Undirected graph
        "chebyshev_K": graph_params.K,
    }

    if train_mask is not None:
        metrics["train_accuracy"] = float(train_acc)
        metrics["n_train_nodes"] = int(np.sum(train_mask))

    if test_mask is not None:
        metrics["n_test_nodes"] = int(np.sum(test_mask))

    OmegaConf.save(metrics, os.path.join(cfg.out_dir, "metrics.yaml"))


# Default configuration
default_config = {
    "task": "tabular_binary",
    "out_dir": "outputs",
    "model": {
        "input_dim": 10,  # Will be overridden
        "n_components": 128,
        "localized": True,
        "learn_centers": True,
        "init_freq_scale": 3.0,
        "geom_hidden": 64,
        "learning_rate": 0.003,
        "max_epochs": 25,
        "batch_size": 256,
        "weight_decay": 0.0,
        "freq_dropout_p": 0.2,
        "grad_clip_norm": 1.0,
        "use_amp": True,
        "l2_freq": 1e-4,
        "group_l1": 1e-3,
        "center_disp": 1e-5,
        "use_seed_freqs": True,
        "seed_frac": 0.2,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_seed": 42,
    },
    "data": {
        "path": "data.npz",
        "class_weight": None,
        "timeseries": {"window": 128, "horizon": 1, "rates": [1, 2, 4], "step": 1},
        "graph": {"K": 3},
    },
}


# Register config with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=default_config)


if __name__ == "__main__":
    main()
