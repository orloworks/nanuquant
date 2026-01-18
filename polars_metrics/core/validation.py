"""Input validation for polars_metrics."""

from __future__ import annotations

import polars as pl

from polars_metrics.exceptions import (
    BenchmarkMismatchError,
    EmptySeriesError,
    InsufficientDataError,
)


def validate_returns(data: pl.Series, *, allow_empty: bool = False) -> None:
    """Validate that data is a valid returns series.

    Parameters
    ----------
    data : pl.Series
        Series to validate.
    allow_empty : bool, default False
        If True, empty series will not raise an error.

    Raises
    ------
    EmptySeriesError
        If data is empty and allow_empty is False.
    TypeError
        If data is not a numeric type.
    """
    if data.is_empty() and not allow_empty:
        raise EmptySeriesError("Returns series is empty")

    if not data.dtype.is_float() and not data.dtype.is_integer():
        raise TypeError(f"Expected numeric dtype, got {data.dtype}")


def validate_min_length(data: pl.Series, min_length: int, metric: str = "") -> None:
    """Validate that data has minimum required length.

    Parameters
    ----------
    data : pl.Series
        Series to validate.
    min_length : int
        Minimum required length.
    metric : str, optional
        Name of the metric for error message.

    Raises
    ------
    InsufficientDataError
        If data length is less than min_length.
    """
    if len(data) < min_length:
        raise InsufficientDataError(
            required=min_length,
            actual=len(data),
            metric=metric,
        )


def validate_benchmark_match(
    strategy: pl.Series,
    benchmark: pl.Series,
) -> None:
    """Validate that strategy and benchmark have matching lengths.

    Parameters
    ----------
    strategy : pl.Series
        Strategy returns.
    benchmark : pl.Series
        Benchmark returns.

    Raises
    ------
    BenchmarkMismatchError
        If lengths don't match.
    """
    if len(strategy) != len(benchmark):
        raise BenchmarkMismatchError(
            strategy_len=len(strategy),
            benchmark_len=len(benchmark),
        )


def validate_positive(value: float, name: str = "value") -> None:
    """Validate that value is positive.

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Name for error message.

    Raises
    ------
    ValueError
        If value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_probability(value: float, name: str = "probability") -> None:
    """Validate that value is a valid probability (0-1).

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Name for error message.

    Raises
    ------
    ValueError
        If value is not in [0, 1].
    """
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
