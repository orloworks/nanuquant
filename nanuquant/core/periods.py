"""Period analysis for nanuquant.

This module provides period-based analysis functions that match QuantStats output,
including monthly returns tables and distribution analysis across timeframes.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from nanuquant.core.utils import to_float_series
from nanuquant.core.validation import validate_returns


def monthly_returns(
    returns: pl.Series,
    *,
    dates: pl.Series | None = None,
    compounded: bool = True,
) -> pl.DataFrame:
    """Create a month-by-year matrix of returns.

    Matches QuantStats monthly_returns implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    dates : pl.Series, optional
        Date series corresponding to returns. If None, assumes daily returns
        starting from 2020-01-01.
    compounded : bool, default True
        If True, compound returns within each month. If False, sum them.

    Returns
    -------
    pl.DataFrame
        DataFrame with months as rows (1-12) and years as columns.
        Values are the monthly returns for each month/year combination.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import date
    >>> returns = pl.Series([0.01, -0.02, 0.015] * 100)
    >>> dates = pl.date_range(date(2020, 1, 1), date(2020, 10, 9), eager=True)
    >>> monthly_returns(returns, dates=dates)
    shape: (10, 2)
    ...
    """
    validate_returns(returns, allow_empty=True)
    if returns.is_empty():
        return pl.DataFrame()

    returns = to_float_series(returns)

    # Generate dates if not provided
    if dates is None:
        from datetime import date, timedelta

        start = date(2020, 1, 1)
        dates = pl.Series(
            "date",
            [start + timedelta(days=i) for i in range(len(returns))],
        )

    # Create DataFrame with returns and dates
    df = pl.DataFrame({
        "date": dates,
        "returns": returns,
    })

    # Extract year and month
    df = df.with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    ])

    # Aggregate returns by month/year
    if compounded:
        # Compound: (1 + r1) * (1 + r2) * ... - 1
        monthly = df.group_by(["year", "month"]).agg(
            ((pl.col("returns") + 1).product() - 1).alias("monthly_return")
        )
    else:
        # Simple sum
        monthly = df.group_by(["year", "month"]).agg(
            pl.col("returns").sum().alias("monthly_return")
        )

    # Pivot to create month-by-year matrix
    result = monthly.pivot(
        on="year",
        index="month",
        values="monthly_return",
    ).sort("month")

    return result


def distribution(
    returns: pl.Series,
    *,
    dates: pl.Series | None = None,
    compounded: bool = True,
) -> dict[str, dict[str, Any]]:
    """Analyze return distribution across multiple time periods.

    Matches QuantStats distribution implementation using IQR method
    for outlier detection.

    Parameters
    ----------
    returns : pl.Series
        Period returns (assumed daily).
    dates : pl.Series, optional
        Date series corresponding to returns. If None, assumes daily returns
        starting from 2020-01-01.
    compounded : bool, default True
        If True, compound returns within each period. If False, sum them.

    Returns
    -------
    dict
        Dictionary with keys for each period (Daily, Weekly, Monthly, Quarterly, Yearly).
        Each period contains:
        - 'values': list of aggregated period returns
        - 'outliers': list of outlier returns (using IQR method)
        - 'mean': mean return for the period
        - 'std': standard deviation of returns
        - 'min': minimum return
        - 'max': maximum return
        - 'count': number of periods

    Notes
    -----
    Outliers are identified using the IQR method:
    - Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are considered outliers.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 100)
    >>> dist = distribution(returns)
    >>> dist.keys()
    dict_keys(['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly'])
    """
    validate_returns(returns, allow_empty=True)
    if returns.is_empty():
        return {}

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

    def aggregate_period(
        data: pl.DataFrame,
        period_col: str,
    ) -> pl.Series:
        """Aggregate returns by period."""
        if compounded:
            agg = data.group_by(period_col).agg(
                ((pl.col("returns") + 1).product() - 1).alias("period_return")
            )
        else:
            agg = data.group_by(period_col).agg(
                pl.col("returns").sum().alias("period_return")
            )
        return agg["period_return"]

    def identify_outliers_iqr(series: pl.Series) -> pl.Series:
        """Identify outliers using IQR method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        if q1 is None or q3 is None:
            return pl.Series([], dtype=pl.Float64)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return series.filter((series < lower_bound) | (series > upper_bound))

    def compute_stats(series: pl.Series) -> dict[str, Any]:
        """Compute statistics for a series."""
        outliers = identify_outliers_iqr(series)
        return {
            "values": series.to_list(),
            "outliers": outliers.to_list(),
            "mean": float(series.mean()) if series.mean() is not None else 0.0,
            "std": float(series.std()) if series.std() is not None else 0.0,
            "min": float(series.min()) if series.min() is not None else 0.0,
            "max": float(series.max()) if series.max() is not None else 0.0,
            "count": len(series),
        }

    result: dict[str, dict[str, Any]] = {}

    # Daily - just the raw returns
    result["Daily"] = compute_stats(returns)

    # Weekly - aggregate by ISO week
    df_weekly = df.with_columns(
        (pl.col("date").dt.year().cast(pl.Utf8) + "-" +
         pl.col("date").dt.week().cast(pl.Utf8)).alias("week")
    )
    weekly_returns = aggregate_period(df_weekly, "week")
    result["Weekly"] = compute_stats(weekly_returns)

    # Monthly
    df_monthly = df.with_columns(
        (pl.col("date").dt.year().cast(pl.Utf8) + "-" +
         pl.col("date").dt.month().cast(pl.Utf8)).alias("month")
    )
    monthly_rets = aggregate_period(df_monthly, "month")
    result["Monthly"] = compute_stats(monthly_rets)

    # Quarterly
    df_quarterly = df.with_columns(
        (pl.col("date").dt.year().cast(pl.Utf8) + "-Q" +
         ((pl.col("date").dt.month() - 1) // 3 + 1).cast(pl.Utf8)).alias("quarter")
    )
    quarterly_returns = aggregate_period(df_quarterly, "quarter")
    result["Quarterly"] = compute_stats(quarterly_returns)

    # Yearly
    df_yearly = df.with_columns(
        pl.col("date").dt.year().alias("year")
    )
    yearly_returns = aggregate_period(df_yearly, "year")
    result["Yearly"] = compute_stats(yearly_returns)

    return result


def compare(
    returns: pl.Series,
    benchmark: pl.Series,
    *,
    dates: pl.Series | None = None,
) -> pl.DataFrame:
    """Compare strategy returns against benchmark across periods.

    Parameters
    ----------
    returns : pl.Series
        Strategy returns.
    benchmark : pl.Series
        Benchmark returns.
    dates : pl.Series, optional
        Date series corresponding to returns. If None, assumes daily returns
        starting from 2020-01-01.

    Returns
    -------
    pl.DataFrame
        DataFrame with period-by-period comparison showing:
        - Strategy return
        - Benchmark return
        - Excess return (strategy - benchmark)
        - Win indicator (1 if strategy > benchmark, 0 otherwise)

    Examples
    --------
    >>> import polars as pl
    >>> strategy = pl.Series([0.01, -0.02, 0.03])
    >>> benchmark = pl.Series([0.005, -0.01, 0.02])
    >>> compare(strategy, benchmark)
    shape: (3, 5)
    ...
    """
    from nanuquant.core.validation import validate_benchmark_match

    validate_returns(returns, allow_empty=True)
    validate_returns(benchmark, allow_empty=True)
    validate_benchmark_match(returns, benchmark)

    if returns.is_empty():
        return pl.DataFrame()

    returns = to_float_series(returns)
    benchmark = to_float_series(benchmark)

    # Generate dates if not provided
    if dates is None:
        from datetime import date, timedelta

        start = date(2020, 1, 1)
        dates = pl.Series(
            "date",
            [start + timedelta(days=i) for i in range(len(returns))],
        )

    return pl.DataFrame({
        "date": dates,
        "strategy": returns,
        "benchmark": benchmark,
        "excess": returns - benchmark,
        "win": (returns > benchmark).cast(pl.Int32),
    })
