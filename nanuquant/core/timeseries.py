"""Timeseries analysis for nanuquant.

This module provides functions that return Series or DataFrames rather than scalars,
designed for tearsheet generation, visualization, and multi-period analysis.

All implementations use native Polars operations for optimal performance.
"""

from __future__ import annotations

from datetime import date, timedelta

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

    # Generate dates if not provided using Polars date_range
    if dates is None:
        start = date(2020, 1, 1)
        dates = pl.date_range(
            start=start,
            end=start + timedelta(days=len(returns) - 1),
            interval="1d",
            eager=True,
        ).alias("date")

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


def drawdown_details(
    returns: pl.Series,
    *,
    top_n: int = 5,
    dates: pl.Series | None = None,
) -> pl.DataFrame:
    """Get detailed information about the top N drawdowns.

    This is the "drawdown table" feature commonly found in fund fact sheets,
    showing start date, end date, depth, and duration of each drawdown period.

    Uses vectorized Polars operations for optimal performance.

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
    n = len(returns)

    # Calculate cumulative wealth and drawdown series using Polars
    cumulative = (1 + returns).cum_prod()
    running_max = cumulative.cum_max()
    drawdown = (cumulative - running_max) / running_max

    # Build a DataFrame for vectorized processing
    df = pl.DataFrame({
        "idx": pl.arange(0, n, eager=True),
        "dd": drawdown,
    })

    # Add date column if provided
    if dates is not None:
        df = df.with_columns(pl.Series("date", dates))

    # Identify drawdown periods using run-length encoding
    # A new drawdown period starts when we transition from dd >= 0 to dd < 0
    df = df.with_columns([
        (pl.col("dd") < 0).alias("in_dd"),
    ])

    # Create group IDs for consecutive drawdown/non-drawdown periods
    df = df.with_columns([
        (pl.col("in_dd") != pl.col("in_dd").shift(1)).fill_null(True).cum_sum().alias("period_id"),
    ])

    # Filter to only drawdown periods (dd < 0)
    dd_periods = df.filter(pl.col("in_dd"))

    if dd_periods.is_empty():
        # No drawdowns found
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

    # Aggregate each drawdown period to find start, valley, end, depth
    period_stats = dd_periods.group_by("period_id").agg([
        pl.col("idx").min().alias("first_dd_idx"),
        pl.col("idx").max().alias("last_dd_idx"),
        pl.col("dd").min().alias("depth"),
        # Find the index where minimum drawdown occurred
        pl.col("idx").filter(pl.col("dd") == pl.col("dd").min()).first().alias("valley_idx"),
    ])

    # Start is one index before first_dd_idx (the peak), clamped to 0
    period_stats = period_stats.with_columns([
        (pl.col("first_dd_idx") - 1).clip(lower_bound=0).alias("start_idx"),
    ])

    # End is one index after last_dd_idx if recovered, otherwise last_dd_idx
    period_stats = period_stats.with_columns([
        pl.when(pl.col("last_dd_idx") < n - 1)
        .then(pl.col("last_dd_idx") + 1)
        .otherwise(pl.col("last_dd_idx"))
        .alias("end_idx"),
        pl.when(pl.col("last_dd_idx") < n - 1)
        .then(True)
        .otherwise(False)
        .alias("recovered"),
    ])

    # Calculate length and recovery
    period_stats = period_stats.with_columns([
        (pl.col("end_idx") - pl.col("start_idx") + 1).alias("length"),
        pl.when(pl.col("recovered"))
        .then(pl.col("end_idx") - pl.col("valley_idx"))
        .otherwise(None)
        .alias("recovery"),
    ])

    # Sort by depth (most negative first) and take top N
    period_stats = period_stats.sort("depth").head(top_n)

    # Map indices to dates if provided
    if dates is not None:
        dates_list = dates.to_list()
        result = pl.DataFrame({
            "start": [dates_list[int(i)] for i in period_stats["start_idx"].to_list()],
            "valley": [dates_list[int(i)] for i in period_stats["valley_idx"].to_list()],
            "end": [dates_list[int(i)] for i in period_stats["end_idx"].to_list()],
            "depth": period_stats["depth"].to_list(),
            "length": period_stats["length"].to_list(),
            "recovery": period_stats["recovery"].to_list(),
        })
    else:
        result = period_stats.select([
            pl.col("start_idx").alias("start"),
            pl.col("valley_idx").alias("valley"),
            pl.col("end_idx").alias("end"),
            pl.col("depth"),
            pl.col("length"),
            pl.col("recovery"),
        ])

    return result


def histogram(
    returns: pl.Series,
    *,
    bins: int = 50,
    density: bool = False,
) -> pl.DataFrame:
    """Calculate histogram of returns for distribution visualization.

    Uses Polars' native histogram implementation for optimal performance.

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
        }).cast({"count": pl.UInt32})

    # Use Polars native hist() for optimal performance
    # hist() returns columns: breakpoint (right edge), category, count
    hist_df = returns.hist(bin_count=bins, include_breakpoint=True)

    total = len(returns)
    bin_width = (max_val - min_val) / bins

    # Build result with proper column names
    # breakpoint is the right edge of each bin
    result = hist_df.select([
        (pl.col("breakpoint") - bin_width).alias("bin_start"),
        pl.col("breakpoint").alias("bin_end"),
        (pl.col("breakpoint") - bin_width / 2).alias("bin_center"),
        pl.col("count").cast(pl.UInt32),
        (pl.col("count") / total).alias("frequency"),
    ])

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
