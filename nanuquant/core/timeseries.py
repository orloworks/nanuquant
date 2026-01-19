"""Timeseries analysis for nanuquant.

This module provides functions that return Series or DataFrames rather than scalars,
designed for tearsheet generation, visualization, and multi-period analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from nanuquant.core.utils import compound_returns, to_float_series
from nanuquant.core.validation import validate_returns


def yearly_returns(
    returns: pl.Series,
    *,
    dates: pl.Series | None = None,
    compounded: bool = True,
) -> pl.DataFrame:
    """Calculate yearly returns for tearsheet generation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    dates : pl.Series, optional
        Date series corresponding to returns. If None, assumes daily returns
        starting from 2020-01-01.
    compounded : bool, default True
        If True, compound returns within each year. If False, sum them.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: year, return.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import date
    >>> returns = pl.Series([0.01, -0.02, 0.015] * 365)
    >>> dates = pl.date_range(date(2020, 1, 1), date(2022, 12, 30), eager=True)[:len(returns)]
    >>> yearly_returns(returns, dates=dates)
    shape: (3, 2)
    ┌──────┬──────────┐
    │ year ┆ return   │
    │ ---  ┆ ---      │
    │ i32  ┆ f64      │
    ╞══════╪══════════╡
    │ 2020 ┆ ...      │
    │ 2021 ┆ ...      │
    │ 2022 ┆ ...      │
    └──────┴──────────┘
    """
    validate_returns(returns, allow_empty=True)
    if returns.is_empty():
        return pl.DataFrame({"year": [], "return": []}).cast({"year": pl.Int32, "return": pl.Float64})

    returns = to_float_series(returns)

    # Generate dates if not provided
    if dates is None:
        from datetime import date, timedelta

        start = date(2020, 1, 1)
        dates = pl.Series(
            "date",
            [start + timedelta(days=i) for i in range(len(returns))],
        )

    df = pl.DataFrame({
        "date": dates,
        "returns": returns,
    })

    # Extract year
    df = df.with_columns(pl.col("date").dt.year().alias("year"))

    # Aggregate returns by year
    if compounded:
        yearly = df.group_by("year").agg(
            ((pl.col("returns") + 1).product() - 1).alias("return")
        )
    else:
        yearly = df.group_by("year").agg(
            pl.col("returns").sum().alias("return")
        )

    return yearly.sort("year")


@dataclass
class DrawdownPeriod:
    """A single drawdown period with details."""

    start_idx: int
    end_idx: int
    valley_idx: int
    depth: float
    length: int
    recovery_length: int | None


def drawdown_details(
    returns: pl.Series,
    *,
    top_n: int = 5,
    dates: pl.Series | None = None,
) -> pl.DataFrame:
    """Get detailed information about the top N drawdowns.

    This is the "drawdown table" feature commonly found in fund fact sheets,
    showing start date, end date, depth, and duration of each drawdown period.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    top_n : int, default 5
        Number of drawdowns to return (sorted by depth, worst first).
    dates : pl.Series, optional
        Date series corresponding to returns. If None, returns use integer indices.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - start: Start date/index of drawdown
        - valley: Date/index of maximum drawdown point
        - end: End date/index of drawdown (when recovered)
        - depth: Maximum drawdown depth (negative value)
        - length: Number of periods in drawdown
        - recovery: Number of periods to recover (None if not yet recovered)

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.10, -0.15, -0.10, 0.05, 0.20, -0.25, 0.10])
    >>> drawdown_details(returns, top_n=3)
    shape: (3, 6)
    ┌───────┬────────┬─────┬────────┬────────┬──────────┐
    │ start ┆ valley ┆ end ┆ depth  ┆ length ┆ recovery │
    │ ---   ┆ ---    ┆ --- ┆ ---    ┆ ---    ┆ ---      │
    │ i64   ┆ i64    ┆ i64 ┆ f64    ┆ i64    ┆ i64      │
    ╞═══════╪════════╪═════╪════════╪════════╪══════════╡
    │ ...   ┆ ...    ┆ ... ┆ ...    ┆ ...    ┆ ...      │
    └───────┴────────┴─────┴────────┴────────┴──────────┘
    """
    validate_returns(returns, allow_empty=True)
    if returns.is_empty():
        if dates is not None:
            return pl.DataFrame({
                "start": pl.Series([], dtype=dates.dtype),
                "valley": pl.Series([], dtype=dates.dtype),
                "end": pl.Series([], dtype=dates.dtype),
                "depth": pl.Series([], dtype=pl.Float64),
                "length": pl.Series([], dtype=pl.Int64),
                "recovery": pl.Series([], dtype=pl.Int64),
            })
        return pl.DataFrame({
            "start": pl.Series([], dtype=pl.Int64),
            "valley": pl.Series([], dtype=pl.Int64),
            "end": pl.Series([], dtype=pl.Int64),
            "depth": pl.Series([], dtype=pl.Float64),
            "length": pl.Series([], dtype=pl.Int64),
            "recovery": pl.Series([], dtype=pl.Int64),
        })

    returns = to_float_series(returns)

    # Calculate cumulative wealth and drawdown series
    cumulative = (1 + returns).cum_prod()
    running_max = cumulative.cum_max()
    drawdown = (cumulative - running_max) / running_max

    # Convert to lists for iteration
    dd_list = drawdown.to_list()
    cum_list = cumulative.to_list()
    n = len(dd_list)

    # Find drawdown periods
    periods: list[DrawdownPeriod] = []
    i = 0

    while i < n:
        # Skip periods at peak (drawdown == 0)
        if dd_list[i] >= 0:
            i += 1
            continue

        # Found start of a drawdown
        start_idx = i - 1 if i > 0 else 0

        # Find the valley (minimum point)
        valley_idx = i
        valley_depth = dd_list[i]

        while i < n and dd_list[i] < 0:
            if dd_list[i] < valley_depth:
                valley_depth = dd_list[i]
                valley_idx = i
            i += 1

        # i is now at recovery point (or end of series)
        if i < n:
            # Recovered
            end_idx = i
            recovery_length = i - valley_idx
        else:
            # Not yet recovered
            end_idx = n - 1
            recovery_length = None

        length = (end_idx if recovery_length else valley_idx) - start_idx + 1

        periods.append(DrawdownPeriod(
            start_idx=start_idx,
            end_idx=end_idx,
            valley_idx=valley_idx,
            depth=valley_depth,
            length=length,
            recovery_length=recovery_length,
        ))

    # Sort by depth (most negative first) and take top N
    periods.sort(key=lambda p: p.depth)
    periods = periods[:top_n]

    # Build result DataFrame
    if dates is not None:
        dates_list = dates.to_list()
        result = pl.DataFrame({
            "start": [dates_list[p.start_idx] for p in periods],
            "valley": [dates_list[p.valley_idx] for p in periods],
            "end": [dates_list[p.end_idx] for p in periods],
            "depth": [p.depth for p in periods],
            "length": [p.length for p in periods],
            "recovery": [p.recovery_length for p in periods],
        })
    else:
        result = pl.DataFrame({
            "start": [p.start_idx for p in periods],
            "valley": [p.valley_idx for p in periods],
            "end": [p.end_idx for p in periods],
            "depth": [p.depth for p in periods],
            "length": [p.length for p in periods],
            "recovery": [p.recovery_length for p in periods],
        })

    return result


def histogram(
    returns: pl.Series,
    *,
    bins: int = 50,
    density: bool = False,
) -> pl.DataFrame:
    """Calculate histogram of returns for distribution visualization.

    Uses Polars' optimized histogram implementation for performance.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    bins : int, default 50
        Number of bins.
    density : bool, default False
        If True, normalize to density (area sums to 1).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - bin_start: Left edge of bin
        - bin_end: Right edge of bin
        - bin_center: Center of bin (useful for plotting)
        - count: Number of values in bin
        - frequency: Relative frequency (count / total)
        - density: Probability density (if density=True)

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 100)
    >>> hist = histogram(returns, bins=20)
    >>> hist.select("bin_center", "count")
    shape: (20, 2)
    ...
    """
    validate_returns(returns, allow_empty=True)
    if returns.is_empty():
        return pl.DataFrame({
            "bin_start": pl.Series([], dtype=pl.Float64),
            "bin_end": pl.Series([], dtype=pl.Float64),
            "bin_center": pl.Series([], dtype=pl.Float64),
            "count": pl.Series([], dtype=pl.UInt32),
            "frequency": pl.Series([], dtype=pl.Float64),
        })

    returns = to_float_series(returns)

    # Get min/max for bin edges
    min_val = returns.min()
    max_val = returns.max()

    if min_val is None or max_val is None:
        return pl.DataFrame({
            "bin_start": pl.Series([], dtype=pl.Float64),
            "bin_end": pl.Series([], dtype=pl.Float64),
            "bin_center": pl.Series([], dtype=pl.Float64),
            "count": pl.Series([], dtype=pl.UInt32),
            "frequency": pl.Series([], dtype=pl.Float64),
        })

    # Handle edge case where all values are the same
    if min_val == max_val:
        return pl.DataFrame({
            "bin_start": [min_val - 0.5],
            "bin_end": [max_val + 0.5],
            "bin_center": [min_val],
            "count": [len(returns)],
            "frequency": [1.0],
        })

    # Calculate bin edges
    bin_width = (max_val - min_val) / bins
    bin_starts = [min_val + i * bin_width for i in range(bins)]
    bin_ends = [min_val + (i + 1) * bin_width for i in range(bins)]
    bin_centers = [(s + e) / 2 for s, e in zip(bin_starts, bin_ends)]

    # Count values in each bin manually (more robust than cut)
    count_list = []
    returns_list = returns.to_list()

    for i in range(bins):
        start = bin_starts[i]
        end = bin_ends[i]
        if i == bins - 1:
            # Last bin includes right edge
            count = sum(1 for r in returns_list if start <= r <= end)
        else:
            # Other bins are [start, end)
            count = sum(1 for r in returns_list if start <= r < end)
        count_list.append(count)

    total = len(returns)
    frequencies = [c / total for c in count_list]

    result = pl.DataFrame({
        "bin_start": bin_starts,
        "bin_end": bin_ends,
        "bin_center": bin_centers,
        "count": count_list,
        "frequency": frequencies,
    }).cast({"count": pl.UInt32})

    if density:
        # Density = frequency / bin_width (so area sums to 1)
        result = result.with_columns(
            (pl.col("frequency") / bin_width).alias("density")
        )

    return result


# Aliases for compound_returns for better discoverability
def cumulative_returns(returns: pl.Series) -> pl.Series:
    """Calculate cumulative compounded returns (equity curve).

    This is an alias for `compound_returns`, named for consistency with
    common financial library conventions.

    Parameters
    ----------
    returns : pl.Series
        Period returns (not prices).

    Returns
    -------
    pl.Series
        Cumulative compounded returns (growth of $1).

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.03])
    >>> cumulative_returns(returns)
    shape: (4,)
    Series: '' [f64]
    [
        0.01
        0.0302
        0.019898
        0.050495
    ]
    """
    return compound_returns(returns)


def equity_curve(returns: pl.Series) -> pl.Series:
    """Calculate equity curve (growth of $1).

    This is an alias for `compound_returns`, named for institutional
    reporting conventions. Returns the cumulative wealth factor.

    Parameters
    ----------
    returns : pl.Series
        Period returns (not prices).

    Returns
    -------
    pl.Series
        Equity curve showing cumulative growth.

    Notes
    -----
    To convert to dollar values, add 1 and multiply by initial investment:
    ``(1 + equity_curve(returns)) * initial_investment``

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.03])
    >>> equity_curve(returns)
    shape: (4,)
    Series: '' [f64]
    [
        0.01
        0.0302
        0.019898
        0.050495
    ]
    """
    return compound_returns(returns)
