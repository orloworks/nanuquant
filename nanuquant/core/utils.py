"""Utility functions for nanuquant."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from nanuquant.exceptions import InvalidFrequencyError
from nanuquant.types import ANNUALIZATION_PERIODS, FrequencyType


def infer_frequency(dates: pl.Series) -> FrequencyType:
    """Infer data frequency from datetime series.

    Parameters
    ----------
    dates : pl.Series
        Datetime series (must have at least 2 values).

    Returns
    -------
    FrequencyType
        Inferred frequency: "D", "W", "M", "H", "min", or "s".

    Raises
    ------
    InvalidFrequencyError
        If frequency cannot be determined.

    Examples
    --------
    >>> dates = pl.Series([datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)])
    >>> infer_frequency(dates)
    'D'
    """
    if len(dates) < 2:
        raise InvalidFrequencyError("Need at least 2 dates to infer frequency")

    # Calculate median time difference
    diffs = dates.diff().drop_nulls()
    median_diff = diffs.median()

    if median_diff is None:
        raise InvalidFrequencyError("Could not calculate median time difference")

    # Convert to timedelta if needed
    if isinstance(median_diff, timedelta):
        td = median_diff
    else:
        # Handle polars duration
        td = timedelta(microseconds=median_diff / 1000)

    # Map to frequency type based on median difference
    seconds = td.total_seconds()

    if seconds < 60:
        return "s"
    elif seconds < 3600:
        return "min"
    elif seconds < 86400:
        return "H"
    elif seconds < 604800:
        return "D"
    elif seconds < 2592000:  # ~30 days
        return "W"
    else:
        return "M"


def get_annualization_factor(
    periods_per_year: int | None = None,
    frequency: FrequencyType | None = None,
) -> float:
    """Get annualization factor for return calculations.

    Parameters
    ----------
    periods_per_year : int, optional
        Explicit periods per year. Takes precedence over frequency.
    frequency : FrequencyType, optional
        Data frequency to use for lookup. Default is "D" (daily).

    Returns
    -------
    float
        Annualization factor (sqrt for volatility, direct for returns).

    Examples
    --------
    >>> get_annualization_factor(frequency="D")
    252.0
    >>> get_annualization_factor(periods_per_year=52)
    52.0
    """
    if periods_per_year is not None:
        return float(periods_per_year)

    freq = frequency or "D"
    return ANNUALIZATION_PERIODS[freq]


def to_float_series(data: pl.Series) -> pl.Series:
    """Convert series to Float64 dtype.

    This function ensures consistent Float64 precision for all financial
    calculations, avoiding potential precision issues with Float32.

    Parameters
    ----------
    data : pl.Series
        Input series (any numeric type).

    Returns
    -------
    pl.Series
        Series with Float64 dtype.

    Notes
    -----
    Always returns Float64 for precision consistency in financial calculations.
    Integer series are safely converted without precision loss.

    Examples
    --------
    >>> import polars as pl
    >>> s = pl.Series([1, 2, 3])  # Int64
    >>> to_float_series(s).dtype
    Float64
    >>> s32 = pl.Series([1.0, 2.0], dtype=pl.Float32)
    >>> to_float_series(s32).dtype
    Float64
    """
    if data.dtype == pl.Float64:
        return data
    return data.cast(pl.Float64)


def safe_divide(
    numerator: float | pl.Series,
    denominator: float | pl.Series,
    default: float = 0.0,
) -> float | pl.Series:
    """Safely divide, returning default for zero denominator.

    Parameters
    ----------
    numerator : float or pl.Series
        Numerator value(s).
    denominator : float or pl.Series
        Denominator value(s).
    default : float, default 0.0
        Value to return when denominator is zero.

    Returns
    -------
    float or pl.Series
        Division result or default.
    """
    if isinstance(denominator, pl.Series):
        return pl.when(denominator != 0).then(numerator / denominator).otherwise(default)
    else:
        if denominator == 0:
            return default
        return numerator / denominator


def compound_returns(returns: pl.Series) -> pl.Series:
    """Calculate cumulative compounded returns.

    Parameters
    ----------
    returns : pl.Series
        Period returns (not prices).

    Returns
    -------
    pl.Series
        Cumulative compounded returns.

    Examples
    --------
    >>> returns = pl.Series([0.01, 0.02, -0.01])
    >>> compound_returns(returns)
    shape: (3,)
    Series: '' [f64]
    [
        0.01
        0.0302
        0.019898
    ]
    """
    return (1 + returns).cum_prod() - 1


def log_returns(returns: pl.Series) -> pl.Series:
    """Convert simple returns to log returns.

    Parameters
    ----------
    returns : pl.Series
        Simple (arithmetic) returns.

    Returns
    -------
    pl.Series
        Log (geometric) returns.
    """
    return (1 + returns).log()


def simple_returns(log_rets: pl.Series) -> pl.Series:
    """Convert log returns to simple returns.

    Parameters
    ----------
    log_rets : pl.Series
        Log returns.

    Returns
    -------
    pl.Series
        Simple (arithmetic) returns.
    """
    return log_rets.exp() - 1
