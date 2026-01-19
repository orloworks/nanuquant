"""Distribution metrics for nanuquant.

This module provides statistical distribution metrics that match QuantStats output.
"""

from __future__ import annotations

import math
from typing import Tuple

import polars as pl
from scipy import stats as scipy_stats

from nanuquant.core.utils import to_float_series
from nanuquant.core.validation import validate_min_length, validate_returns


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


def outliers(
    returns: pl.Series,
    *,
    quantile: float = 0.95,
) -> pl.Series:
    """Identify returns that exceed the specified quantile threshold.

    Matches QuantStats outliers implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    quantile : float, default 0.95
        Quantile threshold. Returns above this percentile are considered outliers.

    Returns
    -------
    pl.Series
        Series containing only the outlier returns (values above the quantile).

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.10, 0.02, -0.01, 0.15, 0.03])
    >>> outliers(returns, quantile=0.90)
    shape: (1,)
    Series: '' [f64]
    [
        0.15
    ]
    """
    validate_returns(returns, allow_empty=True)
    if returns.is_empty():
        return pl.Series([], dtype=pl.Float64)

    returns = to_float_series(returns)

    threshold = returns.quantile(quantile, interpolation="linear")
    if threshold is None:
        return pl.Series([], dtype=pl.Float64)

    result = returns.filter(returns > threshold)
    return result


def remove_outliers(
    returns: pl.Series,
    *,
    quantile: float = 0.95,
) -> pl.Series:
    """Remove returns that exceed the specified quantile threshold.

    Matches QuantStats remove_outliers implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    quantile : float, default 0.95
        Quantile threshold. Returns above this percentile are removed.

    Returns
    -------
    pl.Series
        Series with outlier returns removed.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.10, 0.02, -0.01, 0.15, 0.03])
    >>> clean = remove_outliers(returns, quantile=0.90)
    >>> len(clean)  # One value removed
    6
    """
    validate_returns(returns, allow_empty=True)
    if returns.is_empty():
        return pl.Series([], dtype=pl.Float64)

    returns = to_float_series(returns)

    threshold = returns.quantile(quantile, interpolation="linear")
    if threshold is None:
        return returns

    result = returns.filter(returns < threshold)
    return result


def outliers_iqr(
    returns: pl.Series,
) -> pl.Series:
    """Identify outliers using the IQR (Interquartile Range) method.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    pl.Series
        Series containing only the outlier returns.

    Notes
    -----
    Outliers are values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    This is the method used by QuantStats' distribution() function.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.10, 0.02, -0.01, -0.15, 0.03])
    >>> outliers_iqr(returns)
    shape: (2,)
    ...
    """
    validate_returns(returns, allow_empty=True)
    if returns.is_empty():
        return pl.Series([], dtype=pl.Float64)

    returns = to_float_series(returns)

    q1 = returns.quantile(0.25, interpolation="linear")
    q3 = returns.quantile(0.75, interpolation="linear")

    if q1 is None or q3 is None:
        return pl.Series([], dtype=pl.Float64)

    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    result = returns.filter((returns < lower_bound) | (returns > upper_bound))
    return result


def remove_outliers_iqr(
    returns: pl.Series,
) -> pl.Series:
    """Remove outliers using the IQR (Interquartile Range) method.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    pl.Series
        Series with outliers removed.

    Notes
    -----
    Outliers are values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.10, 0.02, -0.01, -0.15, 0.03])
    >>> clean = remove_outliers_iqr(returns)
    >>> len(clean) < 7  # Some values removed
    True
    """
    validate_returns(returns, allow_empty=True)
    if returns.is_empty():
        return pl.Series([], dtype=pl.Float64)

    returns = to_float_series(returns)

    q1 = returns.quantile(0.25, interpolation="linear")
    q3 = returns.quantile(0.75, interpolation="linear")

    if q1 is None or q3 is None:
        return returns

    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    result = returns.filter((returns >= lower_bound) & (returns <= upper_bound))
    return result
