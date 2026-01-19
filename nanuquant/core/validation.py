"""Input validation for nanuquant.

This module provides validation utilities for returns data with explicit policies
for null handling and dtype enforcement.

Null Handling Policy
--------------------
By default, nanuquant drops null values before calculations. This matches
the behavior of QuantStats and pandas-based libraries where NaN values are
typically excluded. Users can override this behavior by passing `drop_nulls=False`
to validation functions.

Dtype Enforcement
-----------------
All numeric types (int, float) are accepted. Integer types are automatically
cast to Float64 during calculations via `to_float_series()` in utils.py.
"""

from __future__ import annotations

import polars as pl

from nanuquant.exceptions import (
    BenchmarkMismatchError,
    EmptySeriesError,
    InsufficientDataError,
)


def validate_returns(
    data: pl.Series,
    *,
    allow_empty: bool = False,
    drop_nulls: bool = True,
) -> pl.Series:
    """Validate and optionally clean a returns series.

    Parameters
    ----------
    data : pl.Series
        Series to validate.
    allow_empty : bool, default False
        If True, empty series will not raise an error.
    drop_nulls : bool, default True
        If True, null values are dropped from the series before validation.
        This is the default behavior to match QuantStats/pandas conventions.

    Returns
    -------
    pl.Series
        The validated (and optionally cleaned) series.

    Raises
    ------
    EmptySeriesError
        If data is empty (after null removal if drop_nulls=True) and allow_empty is False.
    TypeError
        If data is not a numeric type.

    Notes
    -----
    Null Handling: Polars uses `null` instead of pandas' `NaN` for missing values.
    By default, nulls are dropped to ensure consistent calculations. If you need
    to preserve null positions, set `drop_nulls=False` and handle them manually.

    Examples
    --------
    >>> import polars as pl
    >>> from nanuquant.core.validation import validate_returns
    >>> returns = pl.Series([0.01, None, -0.02, 0.015])
    >>> clean = validate_returns(returns)
    >>> len(clean)  # Null is dropped
    3
    >>> validate_returns(returns, drop_nulls=False)  # Keep nulls
    shape: (4,)
    ...
    """
    # Drop nulls if requested (default behavior)
    if drop_nulls:
        data = data.drop_nulls()

    if data.is_empty() and not allow_empty:
        raise EmptySeriesError("Returns series is empty")

    if not data.dtype.is_float() and not data.dtype.is_integer():
        raise TypeError(f"Expected numeric dtype, got {data.dtype}")

    return data


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
