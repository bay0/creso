"""
Structured logging infrastructure for CReSO.

Provides centralized logging configuration with support for different
output formats, log levels, and structured logging.
"""

from __future__ import annotations

import logging
import logging.config
import time
import functools
from typing import Any, Callable, Dict, Optional, TypeVar, Tuple, MutableMapping
from contextlib import contextmanager

F = TypeVar("F", bound=Callable[..., Any])

# Default logging configuration
DEFAULT_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d %(funcName)s(): %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "structured": {
            "format": "%(asctime)s [%(levelname)8s] %(name)s: %(message)s | %(extra_data)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "creso.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        },
    },
    "loggers": {
        "creso": {"level": "DEBUG", "handlers": ["console", "file"], "propagate": False}
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for structured logging with additional context."""

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Process log message with structured data."""
        extra = kwargs.get("extra", {})
        if hasattr(self, "extra") and self.extra:
            extra.update(self.extra)

        # Format extra data as key=value pairs
        if extra:
            extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
            kwargs["extra"] = {"extra_data": extra_str}
        else:
            kwargs["extra"] = {"extra_data": ""}

        return msg, dict(kwargs)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_type: str = "standard",
    enable_file_logging: bool = False,
) -> None:
    """Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if file logging enabled)
        format_type: Format type ('standard', 'detailed', 'structured')
        enable_file_logging: Whether to enable file logging
    """
    config: Dict[str, Any] = DEFAULT_LOG_CONFIG.copy()

    # Update log level
    config["handlers"]["console"]["level"] = level.upper()
    config["loggers"]["creso"]["level"] = level.upper()

    # Update formatter
    if format_type in config["formatters"]:
        config["handlers"]["console"]["formatter"] = format_type
        config["handlers"]["file"]["formatter"] = format_type

    # Configure file logging
    if enable_file_logging and log_file:
        config["handlers"]["file"]["filename"] = log_file
        config["loggers"]["creso"]["handlers"] = ["console", "file"]
    else:
        config["loggers"]["creso"]["handlers"] = ["console"]

    logging.config.dictConfig(config)


def get_logger(name: str, **extra_context: Any) -> StructuredLoggerAdapter:
    """Get a structured logger with optional context.

    Args:
        name: Logger name
        **extra_context: Additional context to include in all log messages

    Returns:
        Configured logger adapter
    """
    logger = logging.getLogger(name)
    return StructuredLoggerAdapter(logger, extra_context)


def log_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to log function execution time.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger(f"{func.__module__}.{func.__name__}")
        start_time = time.perf_counter()

        try:
            logger.debug(
                f"Starting {func.__name__}",
                extra={"args_count": len(args), "kwargs_count": len(kwargs)},
            )
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time

            logger.info(
                f"Completed {func.__name__}",
                extra={"execution_time": f"{execution_time:.4f}s", "status": "success"},
            )
            return result

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(
                f"Failed {func.__name__}",
                extra={
                    "execution_time": f"{execution_time:.4f}s",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "status": "failed",
                },
            )
            raise

    return wrapper


@contextmanager
def log_context(logger: logging.Logger, operation: str, **context: Any) -> Any:
    """Context manager for logging operation start/end with timing.

    Args:
        logger: Logger instance
        operation: Operation name
        **context: Additional context
    """
    start_time = time.perf_counter()
    logger.info(f"Starting {operation}", extra=context)

    try:
        yield
        execution_time = time.perf_counter() - start_time
        logger.info(
            f"Completed {operation}",
            extra={
                **context,
                "execution_time": f"{execution_time:.4f}s",
                "status": "success",
            },
        )
    except Exception as e:
        execution_time = time.perf_counter() - start_time
        logger.error(
            f"Failed {operation}",
            extra={
                **context,
                "execution_time": f"{execution_time:.4f}s",
                "error": str(e),
                "error_type": type(e).__name__,
                "status": "failed",
            },
        )
        raise


def configure_pytorch_logging() -> None:
    """Configure PyTorch logging to reduce noise."""
    # Reduce PyTorch warnings
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    # Set PyTorch logging levels
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("torch.nn").setLevel(logging.WARNING)
    logging.getLogger("torch.optim").setLevel(logging.WARNING)


# Module-level logger
logger = get_logger(__name__)

# Initialize logging on import
setup_logging()
configure_pytorch_logging()
