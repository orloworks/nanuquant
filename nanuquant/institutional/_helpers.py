"""Helper functions for institutional metrics.

This module provides statistical helper functions for calculating robust
standard errors and autocorrelation adjustments.
"""

from __future__ import annotations

import math

import polars as pl

from nanuquant.core.utils import to_float_series


def _autocorrelation(returns: pl.Series, lag: int = 1) -> float:
    """Calculate autocorrelation at a given lag.

    Parameters
    ----------
    returns : pl.Series
        Return series.
    lag : int, default 1
        Number of periods to lag.

    Returns
    -------
    float
        Autocorrelation coefficient at the specified lag.
        Returns 0.0 if insufficient data or calculation fails.

    Notes
    -----
    Uses the standard formula:
    rho(k) = Cov(r_t, r_{t-k}) / Var(r_t)

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.03, 0.01, -0.02])
    >>> _autocorrelation(returns, lag=1)
    -0.15...
    """
    if len(returns) <= lag:
        return 0.0

    returns = to_float_series(returns)
    n = len(returns)

    mean = returns.mean()
    if mean is None:
        return 0.0

    # Calculate autocorrelation using lagged series
    r_current = returns[lag:]
    r_lagged = returns[:-lag]

    # Covariance at lag k
    cov = ((r_current - mean) * (r_lagged - mean)).sum()
    # Variance
    var = ((returns - mean) ** 2).sum()

    if cov is None or var is None or var == 0:
        return 0.0

    return float(cov / var)


def newey_west_se(
    returns: pl.Series,
    lags: int | None = None,
) -> float:
    """Calculate Newey-West heteroskedasticity and autocorrelation consistent (HAC) standard error.

    Uses Bartlett kernel weights for robust standard error estimation that
    accounts for both heteroskedasticity and serial correlation in returns.

    Parameters
    ----------
    returns : pl.Series
        Return series.
    lags : int, optional
        Number of lags to include. If None, uses the default:
        floor(4 * (n/100)^(2/9)) as recommended by Newey & West (1994).

    Returns
    -------
    float
        Newey-West adjusted standard error of the mean.
        Returns 0.0 if insufficient data.

    Notes
    -----
    The Newey-West estimator uses the Bartlett kernel:
    w(j) = 1 - j / (L + 1)

    The HAC variance estimator is:
    Var_HAC = gamma_0 + 2 * sum_{j=1}^{L} w(j) * gamma_j

    where gamma_j is the j-th autocovariance.

    The standard error is then: SE = sqrt(Var_HAC / n)

    References
    ----------
    Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
    heteroskedasticity and autocorrelation consistent covariance matrix.
    Econometrica, 55(3), 703-708.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.03, 0.01, -0.02] * 50)
    >>> newey_west_se(returns)
    0.002...
    """
    returns = to_float_series(returns)
    n = len(returns)

    if n < 2:
        return 0.0

    # Default lag selection: floor(4 * (n/100)^(2/9))
    if lags is None:
        lags = int(math.floor(4 * (n / 100) ** (2 / 9)))
        lags = max(1, min(lags, n - 2))  # Ensure reasonable bounds

    mean = returns.mean()
    if mean is None:
        return 0.0

    # Calculate demeaned returns
    demeaned = returns - mean

    # Gamma_0: variance (autocovariance at lag 0)
    gamma_0 = (demeaned**2).sum()
    if gamma_0 is None:
        return 0.0
    gamma_0 = float(gamma_0)

    # Sum of weighted autocovariances
    weighted_autocov_sum = 0.0

    for j in range(1, lags + 1):
        if j >= n:
            break

        # Autocovariance at lag j
        cov_j = (demeaned[j:] * demeaned[:-j]).sum()
        if cov_j is None:
            continue

        # Bartlett kernel weight
        weight = 1 - j / (lags + 1)

        # Add twice the weighted autocovariance (for both +j and -j)
        weighted_autocov_sum += 2 * weight * float(cov_j)

    # HAC variance estimate
    var_hac = (gamma_0 + weighted_autocov_sum) / n

    if var_hac <= 0:
        return 0.0

    # Standard error of the mean
    se = math.sqrt(var_hac / n)

    return se


def _expected_max_sharpe(n_trials: int, var_sharpe: float = 1.0) -> float:
    """Calculate expected maximum Sharpe ratio from n independent trials.

    Uses the Euler-Mascheroni constant approximation for the expected
    maximum of n i.i.d. standard normal variables.

    Parameters
    ----------
    n_trials : int
        Number of independent trials/strategies tested.
    var_sharpe : float, default 1.0
        Variance of Sharpe ratio estimates under the null hypothesis.

    Returns
    -------
    float
        Expected maximum Sharpe ratio.

    Notes
    -----
    The expected maximum of n i.i.d. N(0, sigma^2) variables is approximately:
    E[max] = sigma * (sqrt(2 * ln(n)) - (gamma + ln(ln(n) + ln(4*pi))) / (2 * sqrt(2 * ln(n))))

    where gamma is the Euler-Mascheroni constant (~0.5772).

    For large n, this simplifies to approximately:
    E[max] ≈ sigma * sqrt(2 * ln(n))

    References
    ----------
    Bailey, D. H., & Lopez de Prado, M. (2014). The deflated Sharpe ratio:
    correcting for selection bias, backtest overfitting, and non-normality.
    Journal of Portfolio Management, 40(5), 94-107.

    Examples
    --------
    >>> _expected_max_sharpe(100)
    2.32...
    >>> _expected_max_sharpe(1000)
    2.80...
    """
    if n_trials <= 1:
        return 0.0

    # Euler-Mascheroni constant
    gamma = 0.5772156649015329

    # Standard deviation of Sharpe under null
    std_sharpe = math.sqrt(var_sharpe)

    # Calculate expected max using full formula
    ln_n = math.log(n_trials)
    sqrt_2_ln_n = math.sqrt(2 * ln_n)

    # Inner term for correction
    inner = ln_n + math.log(4 * math.pi)
    ln_inner = math.log(inner)
    correction = (gamma + ln_inner) / (2 * sqrt_2_ln_n)

    expected_max = std_sharpe * (sqrt_2_ln_n - correction)

    return max(0.0, expected_max)
