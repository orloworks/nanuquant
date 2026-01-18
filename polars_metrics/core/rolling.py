"""Rolling metrics for polars_metrics.

This module provides rolling (window-based) metrics that match QuantStats output.
"""

from __future__ import annotations

import math

import polars as pl

from polars_metrics.config import get_config
from polars_metrics.core.utils import (
    get_annualization_factor,
    to_float_series,
)
from polars_metrics.core.validation import validate_min_length, validate_returns


def rolling_volatility(
    returns: pl.Series,
    *,
    rolling_period: int = 126,
    periods_per_year: int | None = None,
    annualize: bool = True,
) -> pl.Series:
    """Calculate rolling volatility.

    Matches QuantStats rolling_volatility implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    rolling_period : int, default 126
        Window size for rolling calculation (6 months for daily data).
    periods_per_year : int, optional
        Periods per year for annualization. If None, uses 365 to match QuantStats.
    annualize : bool, default True
        If True, annualize the volatility.

    Returns
    -------
    pl.Series
        Rolling annualized volatility series.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 50)
    >>> rolling_volatility(returns, rolling_period=10)
    shape: (250,)
    ...
    """
    validate_returns(returns, allow_empty=True)
    if returns.is_empty() or len(returns) < rolling_period:
        return pl.Series("rolling_volatility", [], dtype=pl.Float64)

    returns = to_float_series(returns)

    # QuantStats uses 365 by default for annualization in rolling functions
    ann_factor = get_annualization_factor(
        periods_per_year=periods_per_year or 365
    )

    # Calculate rolling std with ddof=1
    rolling_std = returns.rolling_std(window_size=rolling_period, ddof=1)

    if annualize:
        result = rolling_std * math.sqrt(ann_factor)
    else:
        result = rolling_std

    return result.alias("rolling_volatility")


def rolling_sharpe(
    returns: pl.Series,
    *,
    risk_free_rate: float = 0.0,
    rolling_period: int = 126,
    periods_per_year: int | None = None,
    annualize: bool = True,
) -> pl.Series:
    """Calculate rolling Sharpe ratio.

    Matches QuantStats rolling_sharpe implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    risk_free_rate : float, default 0.0
        Annualized risk-free rate.
    rolling_period : int, default 126
        Window size for rolling calculation.
    periods_per_year : int, optional
        Periods per year for annualization. If None, uses 365 to match QuantStats.
    annualize : bool, default True
        If True, annualize the Sharpe ratio.

    Returns
    -------
    pl.Series
        Rolling Sharpe ratio series.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 50)
    >>> rolling_sharpe(returns, rolling_period=10)
    shape: (250,)
    ...
    """
    validate_returns(returns, allow_empty=True)
    if returns.is_empty() or len(returns) < rolling_period:
        return pl.Series("rolling_sharpe", [], dtype=pl.Float64)

    returns = to_float_series(returns)

    # QuantStats uses 365 by default for annualization in rolling functions
    ann_factor = get_annualization_factor(
        periods_per_year=periods_per_year or 365
    )

    # Convert annual risk-free to per-period
    rf_per_period = risk_free_rate / ann_factor

    # Excess returns
    excess_returns = returns - rf_per_period

    # Rolling mean of excess returns
    rolling_mean = excess_returns.rolling_mean(window_size=rolling_period)

    # Rolling std of returns (use original returns, not excess, like QuantStats)
    rolling_std = returns.rolling_std(window_size=rolling_period, ddof=1)

    # Sharpe = mean_excess / std_returns
    # Annualize: multiply by sqrt(ann_factor)
    if annualize:
        result = (rolling_mean / rolling_std) * math.sqrt(ann_factor)
    else:
        result = rolling_mean / rolling_std

    return result.alias("rolling_sharpe")


def rolling_sortino(
    returns: pl.Series,
    *,
    risk_free_rate: float = 0.0,
    rolling_period: int = 126,
    periods_per_year: int | None = None,
    annualize: bool = True,
) -> pl.Series:
    """Calculate rolling Sortino ratio.

    Matches QuantStats rolling_sortino implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    risk_free_rate : float, default 0.0
        Annualized risk-free rate.
    rolling_period : int, default 126
        Window size for rolling calculation.
    periods_per_year : int, optional
        Periods per year for annualization. If None, uses 365 to match QuantStats.
    annualize : bool, default True
        If True, annualize the Sortino ratio.

    Returns
    -------
    pl.Series
        Rolling Sortino ratio series.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 50)
    >>> rolling_sortino(returns, rolling_period=10)
    shape: (250,)
    ...
    """
    validate_returns(returns, allow_empty=True)
    if returns.is_empty() or len(returns) < rolling_period:
        return pl.Series("rolling_sortino", [], dtype=pl.Float64)

    returns = to_float_series(returns)

    # QuantStats uses 365 by default for annualization
    ann_factor = get_annualization_factor(
        periods_per_year=periods_per_year or 365
    )

    # Convert annual risk-free to per-period
    rf_per_period = risk_free_rate / ann_factor

    # Adjust returns for risk-free rate
    adjusted = returns - rf_per_period

    # Rolling mean
    rolling_mean = adjusted.rolling_mean(window_size=rolling_period)

    # For downside deviation, we need custom rolling calculation
    # Use map_batches with a rolling window
    n = len(returns)
    downside_values = []

    for i in range(n):
        if i < rolling_period - 1:
            downside_values.append(None)
        else:
            window = adjusted[i - rolling_period + 1 : i + 1]
            # Get negative returns in window
            negative = window.filter(window < 0)
            if negative.is_empty():
                downside_values.append(0.0)
            else:
                # sqrt(sum(neg^2) / n)
                squared_sum = (negative ** 2).sum()
                downside = math.sqrt(squared_sum / rolling_period)
                downside_values.append(downside)

    downside_series = pl.Series("downside", downside_values)

    # Sortino = mean / downside * sqrt(ann_factor)
    if annualize:
        result = (rolling_mean / downside_series) * math.sqrt(ann_factor)
    else:
        result = rolling_mean / downside_series

    return result.alias("rolling_sortino")


def rolling_beta(
    returns: pl.Series,
    benchmark: pl.Series,
    *,
    rolling_period: int = 126,
) -> pl.Series:
    """Calculate rolling beta relative to benchmark.

    Parameters
    ----------
    returns : pl.Series
        Strategy period returns.
    benchmark : pl.Series
        Benchmark period returns.
    rolling_period : int, default 126
        Window size for rolling calculation.

    Returns
    -------
    pl.Series
        Rolling beta series.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, -0.01, 0.03, 0.01, -0.02] * 50)
    >>> benchmark = pl.Series([0.01, -0.02, 0.02, 0.01, -0.01] * 50)
    >>> rolling_beta(returns, benchmark, rolling_period=10)
    shape: (250,)
    ...
    """
    from polars_metrics.core.validation import validate_benchmark_match

    validate_returns(returns, allow_empty=True)
    validate_returns(benchmark, allow_empty=True)
    validate_benchmark_match(returns, benchmark)

    if returns.is_empty() or len(returns) < rolling_period:
        return pl.Series("rolling_beta", [], dtype=pl.Float64)

    returns = to_float_series(returns)
    benchmark = to_float_series(benchmark)

    n = len(returns)
    beta_values = []

    for i in range(n):
        if i < rolling_period - 1:
            beta_values.append(None)
        else:
            ret_window = returns[i - rolling_period + 1 : i + 1]
            bench_window = benchmark[i - rolling_period + 1 : i + 1]

            mean_ret = ret_window.mean()
            mean_bench = bench_window.mean()

            if mean_ret is None or mean_bench is None:
                beta_values.append(None)
                continue

            # Covariance
            cov = ((ret_window - mean_ret) * (bench_window - mean_bench)).mean()
            # Variance of benchmark
            var_bench = ((bench_window - mean_bench) ** 2).mean()

            if cov is None or var_bench is None or var_bench == 0:
                beta_values.append(None)
            else:
                beta_values.append(float(cov / var_bench))

    return pl.Series("rolling_beta", beta_values)
