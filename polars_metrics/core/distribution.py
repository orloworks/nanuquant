"""Distribution metrics for polars_metrics.

This module provides statistical distribution metrics that match QuantStats output.
"""

from __future__ import annotations

import math
from typing import Tuple

import polars as pl
from scipy import stats as scipy_stats

from polars_metrics.core.utils import to_float_series
from polars_metrics.core.validation import validate_min_length, validate_returns


def skewness(returns: pl.Series) -> float:
    """Calculate skewness of returns.

    Matches QuantStats implementation using scipy.stats.skew.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Skewness of returns. Negative indicates left tail, positive indicates right tail.

    Notes
    -----
    Skewness measures asymmetry of the return distribution.
    - Positive: More frequent small losses, occasional large gains
    - Negative: More frequent small gains, occasional large losses

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.03, -0.01, 0.05])
    >>> skewness(returns)
    0.53...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    if len(returns) < 3:
        return 0.0

    # Use scipy with bias=False to match QuantStats (same as pandas)
    return float(scipy_stats.skew(returns.to_numpy(), bias=False))


def kurtosis(returns: pl.Series) -> float:
    """Calculate excess kurtosis of returns.

    Matches QuantStats implementation using scipy.stats.kurtosis.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Excess kurtosis of returns. Values > 0 indicate fat tails.

    Notes
    -----
    Excess kurtosis measures tail weight relative to normal distribution.
    - Positive (leptokurtic): Fatter tails, more outliers
    - Negative (platykurtic): Thinner tails, fewer outliers
    - Zero (mesokurtic): Similar to normal distribution

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.03, -0.01, 0.05])
    >>> kurtosis(returns)
    -1.38...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    if len(returns) < 4:
        return 0.0

    # Use scipy with bias=False to match QuantStats (same as pandas)
    return float(scipy_stats.kurtosis(returns.to_numpy(), bias=False))


def jarque_bera(returns: pl.Series) -> Tuple[float, float]:
    """Perform Jarque-Bera normality test.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    Tuple[float, float]
        (test_statistic, p_value). Low p-value indicates non-normal distribution.

    Notes
    -----
    Null hypothesis: Data is normally distributed.
    P-value < 0.05 suggests rejection of normality.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.03, -0.01, 0.05] * 20)
    >>> stat, pval = jarque_bera(returns)
    """
    validate_returns(returns)
    if returns.is_empty() or len(returns) < 4:
        return (0.0, 1.0)

    returns = to_float_series(returns)
    stat, pval = scipy_stats.jarque_bera(returns.to_numpy())
    return (float(stat), float(pval))


def shapiro_wilk(returns: pl.Series) -> Tuple[float, float]:
    """Perform Shapiro-Wilk normality test.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    Tuple[float, float]
        (test_statistic, p_value). Low p-value indicates non-normal distribution.

    Notes
    -----
    Null hypothesis: Data is normally distributed.
    P-value < 0.05 suggests rejection of normality.
    Limited to samples of size 3 to 5000.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.03, -0.01, 0.05])
    >>> stat, pval = shapiro_wilk(returns)
    """
    validate_returns(returns)
    if returns.is_empty() or len(returns) < 3:
        return (0.0, 1.0)

    returns = to_float_series(returns)
    # Shapiro-Wilk is limited to 5000 samples
    if len(returns) > 5000:
        returns = returns.head(5000)

    stat, pval = scipy_stats.shapiro(returns.to_numpy())
    return (float(stat), float(pval))


def outlier_win_ratio(returns: pl.Series, quantile: float = 0.99) -> float:
    """Calculate ratio of outlier wins to mean win.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    quantile : float, default 0.99
        Quantile threshold for outliers.

    Returns
    -------
    float
        Ratio of outlier wins (above quantile) to average win.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, 0.03, 0.10, -0.01])
    >>> outlier_win_ratio(returns, quantile=0.95)
    2.5...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    wins = returns.filter(returns > 0)

    if wins.is_empty():
        return 0.0

    mean_win = wins.mean()
    if mean_win is None or mean_win == 0:
        return 0.0

    threshold = wins.quantile(quantile, interpolation="linear")
    if threshold is None:
        return 0.0

    outliers = wins.filter(wins >= threshold)
    if outliers.is_empty():
        return 0.0

    mean_outlier = outliers.mean()
    if mean_outlier is None:
        return 0.0

    return float(mean_outlier / mean_win)


def outlier_loss_ratio(returns: pl.Series, quantile: float = 0.99) -> float:
    """Calculate ratio of outlier losses to mean loss.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    quantile : float, default 0.99
        Quantile threshold for outliers (applied to absolute losses).

    Returns
    -------
    float
        Ratio of outlier losses (absolute) to absolute average loss.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, -0.03, -0.15, 0.02])
    >>> outlier_loss_ratio(returns, quantile=0.95)
    2.0...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    losses = returns.filter(returns < 0).abs()

    if losses.is_empty():
        return 0.0

    mean_loss = losses.mean()
    if mean_loss is None or mean_loss == 0:
        return 0.0

    threshold = losses.quantile(quantile, interpolation="linear")
    if threshold is None:
        return 0.0

    outliers = losses.filter(losses >= threshold)
    if outliers.is_empty():
        return 0.0

    mean_outlier = outliers.mean()
    if mean_outlier is None:
        return 0.0

    return float(mean_outlier / mean_loss)


def expected_return(returns: pl.Series) -> float:
    """Calculate expected return (mean).

    Matches QuantStats expected_return function.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Expected (mean) return.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.03])
    >>> expected_return(returns)
    0.0125
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    result = returns.mean()
    return float(result) if result is not None else 0.0


def geometric_mean(returns: pl.Series) -> float:
    """Calculate geometric mean of returns.

    Matches QuantStats geometric_mean function.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Geometric mean return per period.

    Notes
    -----
    Formula: (product(1 + returns))^(1/n) - 1

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.03])
    >>> geometric_mean(returns)
    0.0123...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    n = len(returns)

    # Calculate product of (1 + returns)
    product = (1 + returns).product()

    if product is None or product <= 0:
        return 0.0

    return float(product ** (1.0 / n) - 1)
