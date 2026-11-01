"""Distribution metrics for nanuquant.

This module provides statistical distribution metrics that match QuantStats output.
"""

from __future__ import annotations

import math
import warnings

import polars as pl
from scipy import stats as scipy_stats

from nanuquant.core.utils import to_float_series
from nanuquant.core.validation import validate_returns


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
    returns = validate_returns(returns)
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
    returns = validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    if len(returns) < 4:
        return 0.0

    # Use scipy with bias=False to match QuantStats (same as pandas)
    return float(scipy_stats.kurtosis(returns.to_numpy(), bias=False))


def jarque_bera(returns: pl.Series) -> tuple[float, float]:
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
    returns = validate_returns(returns)
    if returns.is_empty() or len(returns) < 4:
        return (0.0, 1.0)

    returns = to_float_series(returns)
    stat, pval = scipy_stats.jarque_bera(returns.to_numpy())
    return (float(stat), float(pval))


def shapiro_wilk(returns: pl.Series) -> tuple[float, float]:
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
    returns = validate_returns(returns)
    if returns.is_empty() or len(returns) < 3:
        return (0.0, 1.0)

    returns = to_float_series(returns)
    # Shapiro-Wilk is limited to 5000 samples
    if len(returns) > 5000:
        warnings.warn(
            f"Shapiro-Wilk limited to 5000 samples; truncating from {len(returns)}. "
            "Consider jarque_bera() for larger datasets.",
            UserWarning,
            stacklevel=2,
        )
        returns = returns.head(5000)

    stat, pval = scipy_stats.shapiro(returns.to_numpy())
    return (float(stat), float(pval))


def outlier_win_ratio(returns: pl.Series, quantile: float = 0.99) -> float:
    """Calculate the outlier win ratio.

    Matches QuantStats ``outlier_win_ratio``: the ``quantile`` return of the whole
    series divided by the mean of the non-negative returns. It measures how far the
    best periods stand above a typical up period.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    quantile : float, default 0.99
        Upper-tail quantile of all returns used as the outlier level.

    Returns
    -------
    float
        Outlier win ratio. 0.0 if there are no non-negative returns or their mean is 0.
    """
    returns = validate_returns(returns)
    if returns.is_empty():
        return 0.0
    returns = to_float_series(returns)
    level = returns.quantile(quantile, interpolation="linear")
    up = returns.filter(returns >= 0)
    mean_up = up.mean() if not up.is_empty() else None
    if level is None or mean_up is None or mean_up == 0:
        return 0.0
    return float(level / mean_up)


def outlier_loss_ratio(returns: pl.Series, quantile: float = 0.01) -> float:
    """Calculate the outlier loss ratio.

    Matches QuantStats ``outlier_loss_ratio``: the ``quantile`` return of the whole
    series divided by the mean of the negative returns. It measures how far the worst
    periods stand below a typical down period.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    quantile : float, default 0.01
        Lower-tail quantile of all returns used as the outlier level.

    Returns
    -------
    float
        Outlier loss ratio (positive when the tail quantile is a loss). 0.0 if there
        are no negative returns.
    """
    returns = validate_returns(returns)
    if returns.is_empty():
        return 0.0
    returns = to_float_series(returns)
    level = returns.quantile(quantile, interpolation="linear")
    down = returns.filter(returns < 0)
    mean_down = down.mean() if not down.is_empty() else None
    if level is None or mean_down is None or mean_down == 0:
        return 0.0
    return float(level / mean_down)


def expected_return(returns: pl.Series) -> float:
    """Calculate the expected (geometric mean) return per period.

    Matches QuantStats ``expected_return``: ``prod(1 + r) ** (1 / n) - 1``. For the
    arithmetic mean use :func:`nanuquant.avg_return` with ``exclude_zeros=False``.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Geometric mean return per period.
    """
    return geometric_mean(returns)


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
    returns = validate_returns(returns)
    if returns.is_empty():
        return 0.0

    growth = 1 + to_float_series(returns)

    if (growth == 0).any():
        return -1.0  # total loss
    if (growth < 0).sum() % 2 == 1:
        return float("nan")  # wealth below zero has no real geometric mean

    # Mean log growth instead of product ** (1 / n): the product of a long series
    # underflows to 0 (read as a total loss) or overflows to inf.
    log_mean = growth.abs().log().mean()
    if log_mean is None:
        return 0.0
    return float(math.expm1(float(log_mean)))


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
    returns = validate_returns(returns, allow_empty=True)
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
    returns = validate_returns(returns, allow_empty=True)
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
    returns = validate_returns(returns, allow_empty=True)
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
    returns = validate_returns(returns, allow_empty=True)
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
