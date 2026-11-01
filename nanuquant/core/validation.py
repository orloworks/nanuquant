"""Input validation for nanuquant.

This module provides validation utilities for returns data with explicit policies
for null handling and dtype enforcement.

Missing-Data Policy
-------------------
A missing observation is either a Polars ``null`` or a float ``NaN``; both are treated
the same way. QuantStats fills missing returns with 0, which silently turns a missing
day into a flat day and biases every count, mean and annualization. nanuquant instead:

- **single-series metrics** drop missing observations (:func:`validate_returns`);
- **paired metrics** (strategy vs benchmark, two assets) keep only the rows where both
  sides are present (:func:`align_pair`), so the series stay aligned;
- **dated metrics** drop a missing return together with its date (:func:`align_dates`);
- **positional outputs** (rolling metrics, drawdown series) keep their length and are
  null wherever the input is missing (:func:`normalize_missing`).

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
        If True, missing values (null or NaN) are dropped before validation. If False,
        they are kept as null (NaN is converted to null).

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
    Missing values: both ``null`` and float ``NaN`` are missing (see the module
    docstring). By default they are dropped, never treated as zero returns. To keep
    their positions, set ``drop_nulls=False`` and handle the nulls yourself.

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
    data = normalize_missing(data)
    # Drop missing observations if requested (default behavior)
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


def normalize_missing(data: pl.Series) -> pl.Series:
    """Return ``data`` with float ``NaN`` converted to ``null``, keeping every position.

    Parameters
    ----------
    data : pl.Series
        Series to normalise.

    Returns
    -------
    pl.Series
        Same length as ``data``; ``NaN`` replaced by ``null``.
    """
    if data.dtype.is_float():
        return data.fill_nan(None)
    return data


def align_pair(
    first: pl.Series,
    second: pl.Series,
    *,
    allow_empty: bool = False,
) -> tuple[pl.Series, pl.Series]:
    """Validate two same-length series and keep only rows where both are present.

    Parameters
    ----------
    first, second : pl.Series
        Series observed on the same periods (e.g. strategy and benchmark returns).
    allow_empty : bool, default False
        If True, an empty result is returned instead of raising.

    Returns
    -------
    tuple[pl.Series, pl.Series]
        The two series restricted to the rows where neither is missing.

    Raises
    ------
    BenchmarkMismatchError
        If the inputs have different lengths.
    EmptySeriesError
        If no row has both values present and ``allow_empty`` is False.
    TypeError
        If either series is not numeric.
    """
    validate_benchmark_match(first, second)
    a = normalize_missing(first)
    b = normalize_missing(second)
    for s in (a, b):
        if not s.dtype.is_float() and not s.dtype.is_integer() and s.dtype != pl.Null:
            raise TypeError(f"Expected numeric dtype, got {s.dtype}")
    keep = a.is_not_null() & b.is_not_null()
    a, b = a.filter(keep), b.filter(keep)
    if a.is_empty() and not allow_empty:
        raise EmptySeriesError("No periods with both series present")
    return a, b


def align_dates(
    returns: pl.Series,
    dates: pl.Series | None,
    *,
    allow_empty: bool = False,
) -> tuple[pl.Series, pl.Series | None]:
    """Drop missing returns together with their dates.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    dates : pl.Series or None
        Dates for each return, same length as ``returns``; None if not supplied.
    allow_empty : bool, default False
        If True, an empty result is returned instead of raising.

    Returns
    -------
    tuple[pl.Series, pl.Series | None]
        Returns and dates restricted to the periods with a return.

    Raises
    ------
    ValueError
        If ``dates`` is given and its length differs from ``returns``.
    """
    if dates is not None and len(dates) != len(returns):
        raise ValueError(f"dates length ({len(dates)}) must match returns length ({len(returns)})")
    r = normalize_missing(returns)
    keep = r.is_not_null()
    r = r.filter(keep)
    d = dates.filter(keep) if dates is not None else None
    if r.is_empty() and not allow_empty:
        raise EmptySeriesError("Returns series is empty")
    if not r.dtype.is_float() and not r.dtype.is_integer() and not r.is_empty():
        raise TypeError(f"Expected numeric dtype, got {r.dtype}")
    return r, d
