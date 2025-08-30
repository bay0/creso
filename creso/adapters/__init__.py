"""
CReSO Adapters

Specialized adapters for different data modalities.
"""

try:
    from .timeseries import TimeSeriesCReSOClassifier, make_multirate_windows

    __all__ = ["TimeSeriesCReSOClassifier", "make_multirate_windows"]
except ImportError:
    __all__ = []

try:
    from .graph import GraphNodeCReSOClassifier  # noqa: F401

    __all__.append("GraphNodeCReSOClassifier")
except ImportError:
    pass
