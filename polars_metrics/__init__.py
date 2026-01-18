"""Polars Metrics - Native Polars metrics library for quantitative finance.

A complete QuantStats replacement with native Polars implementation.
"""

from polars_metrics._version import __version__
from polars_metrics.config import DEFAULT_CONFIG, MetricsConfig, get_config, set_config
from polars_metrics.exceptions import (
    BenchmarkMismatchError,
    EmptySeriesError,
    InsufficientDataError,
    InvalidFrequencyError,
    MetricsError,
)

__all__ = [
    "__version__",
    # Config
    "MetricsConfig",
    "DEFAULT_CONFIG",
    "get_config",
    "set_config",
    # Exceptions
    "MetricsError",
    "EmptySeriesError",
    "InsufficientDataError",
    "BenchmarkMismatchError",
    "InvalidFrequencyError",
]
