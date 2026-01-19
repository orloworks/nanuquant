"""Custom exceptions for nanuquant."""

from __future__ import annotations


class MetricsError(Exception):
    """Base exception for nanuquant."""


class EmptySeriesError(MetricsError):
    """Raised when returns series is empty."""

    def __init__(self, message: str = "Returns series is empty") -> None:
        super().__init__(message)


class InsufficientDataError(MetricsError):
    """Raised when not enough data for calculation."""

    def __init__(self, required: int, actual: int, metric: str = "") -> None:
        msg = f"Insufficient data for {metric}: requires {required}, got {actual}"
        super().__init__(msg)
        self.required = required
        self.actual = actual
        self.metric = metric


class BenchmarkMismatchError(MetricsError):
    """Raised when strategy and benchmark lengths differ."""

    def __init__(self, strategy_len: int, benchmark_len: int) -> None:
        msg = f"Strategy length ({strategy_len}) != benchmark length ({benchmark_len})"
        super().__init__(msg)
        self.strategy_len = strategy_len
        self.benchmark_len = benchmark_len


class InvalidFrequencyError(MetricsError):
    """Raised when frequency cannot be determined or is invalid."""

    def __init__(self, message: str = "Cannot determine data frequency") -> None:
        super().__init__(message)
