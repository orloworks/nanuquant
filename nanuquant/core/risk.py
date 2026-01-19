"""Risk metrics for nanuquant.

This module provides risk-based metrics that match QuantStats output.
"""

from __future__ import annotations

import math
from scipy import stats as scipy_stats

import polars as pl

from nanuquant.config import get_config
from nanuquant.core.utils import (
    compound_returns,
    get_annualization_factor,
    to_float_series,
)
from nanuquant.core.validation import validate_min_length, validate_returns
from nanuquant.types import VAR_SIGMA_MAP


def volatility(
    returns: pl.Series,
    *,
    periods_per_year: int | None = None,
    annualize: bool = True,
) -> float:
    """Calculate return volatility (standard deviation).

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    periods_per_year : int, optional
        Number of periods per year for annualization. If None, uses config default.
    annualize : bool, default True
        If True, annualize the volatility using sqrt(periods_per_year).

    Returns
    -------
    float
        Annualized (or raw) volatility of returns.

    Notes
    -----
    Formula: std(returns) * sqrt(periods_per_year)

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02])
    >>> volatility(returns, periods_per_year=252)
    0.248...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    config = get_config()

    # Use sample std dev (ddof=1)
    std = returns.std(ddof=config.ddof)
    if std is None:
        return 0.0

    if annualize:
        ann_factor = get_annualization_factor(
            periods_per_year=periods_per_year or config.periods_per_year
        )
        return float(std * math.sqrt(ann_factor))

    return float(std)


def var(
    returns: pl.Series,
    *,
    sigma: float = 1.0,
    confidence: float = 0.95,
) -> float:
    """Calculate Value at Risk (VaR) using parametric method.

    Matches QuantStats implementation using normal distribution.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    sigma : float, default 1.0
        Sigma multiplier for standard deviation.
    confidence : float, default 0.95
        Confidence level (e.g., 0.95 for 95%).

    Returns
    -------
    float
        VaR as a negative number representing the return at the given confidence.

    Notes
    -----
    Uses parametric VaR: norm.ppf(1 - confidence, mean, sigma * std)

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([-0.05, -0.02, 0.01, 0.03, -0.01])
    >>> var(returns, confidence=0.95)
    -0.058...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)

    mean_ret = returns.mean()
    std_ret = returns.std()

    if mean_ret is None or std_ret is None:
        return 0.0

    # Use scipy's norm.ppf to match QuantStats exactly
    return float(scipy_stats.norm.ppf(1 - confidence, mean_ret, sigma * std_ret))


def cvar(
    returns: pl.Series,
    *,
    sigma: float = 1.0,
    confidence: float = 0.95,
) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall).

    CVaR is the expected loss given that loss exceeds VaR.
    Matches QuantStats implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    sigma : float, default 1.0
        Sigma multiplier for VaR calculation.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    float
        CVaR as a negative number representing expected return in tail.

    Notes
    -----
    CVaR = mean(returns[returns < VaR])

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([-0.10, -0.05, -0.02, 0.01, 0.03])
    >>> cvar(returns, confidence=0.95)
    -0.10
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)

    # Get the VaR threshold using parametric method
    var_threshold = var(returns, sigma=sigma, confidence=confidence)

    # Calculate expected loss in the tail (values < VaR)
    tail_losses = returns.filter(returns < var_threshold)

    if tail_losses.is_empty():
        # If no values in tail, return VaR
        return float(var_threshold)

    expected_shortfall = tail_losses.mean()
    if expected_shortfall is None:
        return float(var_threshold)

    return float(expected_shortfall)


def to_drawdown_series(returns: pl.Series) -> pl.Series:
    """Convert returns to drawdown series.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    pl.Series
        Drawdown series (always negative or zero values).

    Notes
    -----
    Drawdown at time t = (cumulative_wealth[t] - running_max[t]) / running_max[t]

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.10, -0.05, 0.02, -0.08])
    >>> to_drawdown_series(returns)
    shape: (4,)
    Series: '' [f64]
    [
        0.0
        -0.045454...
        -0.027272...
        -0.105454...
    ]
    """
    validate_returns(returns, allow_empty=True)
    if returns.is_empty():
        return pl.Series("drawdown", [], dtype=pl.Float64)

    returns = to_float_series(returns)

    # Calculate cumulative wealth (1 + cumulative return)
    cumulative = (1 + returns).cum_prod()

    # Running maximum
    running_max = cumulative.cum_max()

    # Drawdown as percentage from peak
    drawdown = (cumulative - running_max) / running_max

    return drawdown.alias("drawdown")


def max_drawdown(returns: pl.Series) -> float:
    """Calculate maximum drawdown.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Maximum drawdown as a negative number (e.g., -0.30 means 30% drawdown).

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.10, -0.05, 0.02, -0.15, 0.05])
    >>> max_drawdown(returns)
    -0.134...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    dd_series = to_drawdown_series(returns)
    result = dd_series.min()

    if result is None:
        return 0.0

    return float(result)


def ulcer_index(returns: pl.Series) -> float:
    """Calculate the Ulcer Index (drawdown severity measure).

    Matches QuantStats implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Ulcer Index (always positive).

    Notes
    -----
    Ulcer Index = sqrt(sum(drawdown^2) / (n - 1))
    Uses sample formula (n-1 denominator) to match QuantStats.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.015, -0.03, 0.02])
    >>> ulcer_index(returns)
    0.015...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    n = len(returns)
    if n <= 1:
        return 0.0

    dd_series = to_drawdown_series(returns)

    # Square the drawdowns and sum
    dd_squared_sum = (dd_series ** 2).sum()

    if dd_squared_sum is None:
        return 0.0

    # Use (n - 1) denominator to match QuantStats
    return float(math.sqrt(dd_squared_sum / (n - 1)))


def downside_deviation(
    returns: pl.Series,
    *,
    mar: float | None = None,
    periods_per_year: int | None = None,
    annualize: bool = True,
) -> float:
    """Calculate downside deviation (semi-deviation below MAR).

    Matches QuantStats Sortino calculation formula.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    mar : float, optional
        Minimum Acceptable Return (per period). If None, uses config default (0).
    periods_per_year : int, optional
        Periods per year for annualization.
    annualize : bool, default True
        Whether to annualize the result.

    Returns
    -------
    float
        Downside deviation.

    Notes
    -----
    Formula: sqrt(sum(min(returns, 0)^2) / n)
    This matches the Red Rock Capital Sortino paper formula used by QuantStats.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.015, -0.03, 0.02])
    >>> downside_deviation(returns, mar=0.0)
    0.18...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    config = get_config()

    target = mar if mar is not None else config.mar
    n = len(returns)

    # Get returns below target and square them
    excess = returns - target
    negative_returns = excess.filter(excess < 0)

    if negative_returns.is_empty():
        return 0.0

    # Sum of squared negative returns divided by total count
    squared_sum = (negative_returns ** 2).sum()
    if squared_sum is None:
        return 0.0

    dd = math.sqrt(squared_sum / n)

    if annualize:
        ann_factor = get_annualization_factor(
            periods_per_year=periods_per_year or config.periods_per_year
        )
        return float(dd * math.sqrt(ann_factor))

    return float(dd)
