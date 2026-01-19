"""Rolling metrics for nanuquant.

This module provides rolling (window-based) metrics that match QuantStats output.

Note on Annualization Periods
-----------------------------
Rolling metrics in this module default to 365 periods per year to match
QuantStats' rolling function behavior. This differs from the global config
default of 252 trading days used by core metrics like `sharpe()` and `volatility()`.

The rationale:
- QuantStats rolling functions use calendar days (365) for consistency with
  their datetime-indexed calculations
- Core metrics use trading days (252) as the standard for daily returns

To use trading days for rolling metrics, explicitly pass `periods_per_year=252`:

    >>> rolling_sharpe(returns, periods_per_year=252)

Or use the global config by passing `periods_per_year=get_config().periods_per_year`.
"""

from __future__ import annotations

import math

import polars as pl

from nanuquant.config import get_config
from nanuquant.core.utils import (
    get_annualization_factor,
    to_float_series,
)
from nanuquant.core.validation import validate_min_length, validate_returns


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

    This implementation uses native Polars vectorized operations for optimal
    performance on large datasets, avoiding Python loops.

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

    Notes
    -----
    Downside deviation is calculated as sqrt(sum(negative_returns^2) / n),
    where n is the window size. This matches the QuantStats formula.

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

    # Rolling mean of adjusted returns
    rolling_mean = adjusted.rolling_mean(window_size=rolling_period)

    # Vectorized downside deviation calculation:
    # For each value: if negative, square it; otherwise 0
    # Then rolling sum / n, then sqrt
    # clip(upper_bound=0) sets positive values to 0, pow(2) squares the negatives
    negative_squared = adjusted.clip(upper_bound=0).pow(2)
    rolling_neg_sq_sum = negative_squared.rolling_sum(window_size=rolling_period)
    downside_series = (rolling_neg_sq_sum / rolling_period).sqrt()

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

    This implementation uses native Polars vectorized operations for optimal
    performance on large datasets, avoiding Python loops.

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

    Notes
    -----
    Beta is calculated as Cov(returns, benchmark) / Var(benchmark).

    Using the identity:
        Cov(X, Y) = E[XY] - E[X]E[Y]
        Var(Y) = E[Y^2] - E[Y]^2

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, -0.01, 0.03, 0.01, -0.02] * 50)
    >>> benchmark = pl.Series([0.01, -0.02, 0.02, 0.01, -0.01] * 50)
    >>> rolling_beta(returns, benchmark, rolling_period=10)
    shape: (250,)
    ...
    """
    from nanuquant.core.validation import validate_benchmark_match

    validate_returns(returns, allow_empty=True)
    validate_returns(benchmark, allow_empty=True)
    validate_benchmark_match(returns, benchmark)

    if returns.is_empty() or len(returns) < rolling_period:
        return pl.Series("rolling_beta", [], dtype=pl.Float64)

    returns = to_float_series(returns)
    benchmark = to_float_series(benchmark)

    # Vectorized rolling beta calculation using covariance formula:
    # Beta = Cov(X, Y) / Var(Y)
    # Cov(X, Y) = E[XY] - E[X]E[Y]
    # Var(Y) = E[Y^2] - E[Y]^2

    # Rolling means
    rolling_mean_ret = returns.rolling_mean(window_size=rolling_period)
    rolling_mean_bench = benchmark.rolling_mean(window_size=rolling_period)

    # Rolling mean of product (for covariance)
    rolling_mean_product = (returns * benchmark).rolling_mean(window_size=rolling_period)

    # Rolling mean of benchmark squared (for variance)
    rolling_mean_bench_sq = (benchmark ** 2).rolling_mean(window_size=rolling_period)

    # Covariance: E[XY] - E[X]E[Y]
    rolling_cov = rolling_mean_product - rolling_mean_ret * rolling_mean_bench

    # Variance of benchmark: E[Y^2] - E[Y]^2
    rolling_var_bench = rolling_mean_bench_sq - rolling_mean_bench ** 2

    # Beta = Cov / Var (will be null where var is 0 due to division)
    result = rolling_cov / rolling_var_bench

    return result.alias("rolling_beta")


def rolling_greeks(
    returns: pl.Series,
    benchmark: pl.Series,
    *,
    rolling_period: int = 126,
    periods_per_year: int | None = None,
) -> pl.DataFrame:
    """Calculate rolling alpha and beta relative to benchmark.

    Matches QuantStats rolling_greeks implementation. Returns both alpha and beta
    as time series in a DataFrame.

    This implementation uses native Polars vectorized operations for optimal
    performance on large datasets, avoiding Python loops.

    Parameters
    ----------
    returns : pl.Series
        Strategy period returns.
    benchmark : pl.Series
        Benchmark period returns.
    rolling_period : int, default 126
        Window size for rolling calculation (6 months for daily data).
    periods_per_year : int, optional
        Periods per year for annualization. If None, uses 365 to match QuantStats.

    Returns
    -------
    pl.DataFrame
        DataFrame with two columns:
        - 'rolling_alpha': Rolling annualized alpha
        - 'rolling_beta': Rolling beta

    Notes
    -----
    Beta is calculated as Cov(returns, benchmark) / Var(benchmark).
    Alpha is calculated as: mean(returns) - beta * mean(benchmark), annualized.

    Using the identity:
        Cov(X, Y) = E[XY] - E[X]E[Y]
        Var(Y) = E[Y^2] - E[Y]^2

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, -0.01, 0.03, 0.01, -0.02] * 50)
    >>> benchmark = pl.Series([0.01, -0.02, 0.02, 0.01, -0.01] * 50)
    >>> greeks = rolling_greeks(returns, benchmark, rolling_period=10)
    >>> greeks.columns
    ['rolling_alpha', 'rolling_beta']
    """
    from nanuquant.core.validation import validate_benchmark_match

    validate_returns(returns, allow_empty=True)
    validate_returns(benchmark, allow_empty=True)
    validate_benchmark_match(returns, benchmark)

    if returns.is_empty() or len(returns) < rolling_period:
        return pl.DataFrame({
            "rolling_alpha": pl.Series([], dtype=pl.Float64),
            "rolling_beta": pl.Series([], dtype=pl.Float64),
        })

    returns = to_float_series(returns)
    benchmark = to_float_series(benchmark)

    # QuantStats uses 365 by default for annualization
    ann_factor = get_annualization_factor(
        periods_per_year=periods_per_year or 365
    )

    # Rolling means
    rolling_mean_ret = returns.rolling_mean(window_size=rolling_period)
    rolling_mean_bench = benchmark.rolling_mean(window_size=rolling_period)

    # Rolling mean of product (for covariance)
    rolling_mean_product = (returns * benchmark).rolling_mean(window_size=rolling_period)

    # Rolling mean of benchmark squared (for variance)
    rolling_mean_bench_sq = (benchmark ** 2).rolling_mean(window_size=rolling_period)

    # Covariance: E[XY] - E[X]E[Y]
    rolling_cov = rolling_mean_product - rolling_mean_ret * rolling_mean_bench

    # Variance of benchmark: E[Y^2] - E[Y]^2
    rolling_var_bench = rolling_mean_bench_sq - rolling_mean_bench ** 2

    # Beta = Cov / Var
    beta = rolling_cov / rolling_var_bench

    # Alpha = mean_returns - beta * mean_benchmark, annualized
    # Annualize by multiplying by periods_per_year
    alpha = (rolling_mean_ret - beta * rolling_mean_bench) * ann_factor

    return pl.DataFrame({
        "rolling_alpha": alpha,
        "rolling_beta": beta,
    })
