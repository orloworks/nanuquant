"""Extended Value at Risk metrics.

This module provides advanced VaR measures that account for non-normality
of return distributions, including Cornish-Fisher and Entropic VaR.

References
----------
- Cornish, E. A., & Fisher, R. A. (1937). "Moments and Cumulants in the
  Specification of Distributions"
- Ahmadi-Javid, A. (2012). "Entropic Value-at-Risk: A New Coherent Risk Measure"
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy import optimize, stats as scipy_stats

from nanuquant.core.utils import to_float_series
from nanuquant.core.validation import validate_min_length, validate_returns


def cornish_fisher_var(
    returns: pl.Series,
    confidence: float = 0.95,
) -> float:
    """Calculate Cornish-Fisher adjusted Value at Risk.

    The Cornish-Fisher expansion adjusts the VaR quantile to account for
    skewness and excess kurtosis in the return distribution, providing
    a more accurate estimate for non-normal returns.

    Parameters
    ----------
    returns : pl.Series
        Series of returns.
    confidence : float, default 0.95
        Confidence level (e.g., 0.95 for 95% VaR).

    Returns
    -------
    float
        Cornish-Fisher adjusted VaR (as a positive loss amount).
        Higher values indicate greater potential loss.

    Notes
    -----
    The Cornish-Fisher expansion modifies the normal quantile:

    z_cf = z + (z² - 1) * S/6 + (z³ - 3z) * K/24 - (2z³ - 5z) * S²/36

    Where:
    - z is the standard normal quantile
    - S is skewness
    - K is excess kurtosis

    VaR_cf = -(μ + z_cf * σ)

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> returns = pl.Series(np.random.normal(0, 0.02, 500))
    >>> cf_var = cornish_fisher_var(returns, confidence=0.95)
    >>> cf_var > 0
    True

    References
    ----------
    Cornish, E. A., & Fisher, R. A. (1937)
    """
    validate_returns(returns)
    validate_min_length(returns, 4, "cornish_fisher_var")

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    returns = to_float_series(returns)

    # Calculate moments
    mu = returns.mean()
    sigma = returns.std(ddof=1)

    if mu is None or sigma is None or sigma == 0:
        return 0.0

    mu = float(mu)
    sigma = float(sigma)

    # Skewness and kurtosis
    skew = float(scipy_stats.skew(returns.to_numpy(), bias=False))
    kurt = float(scipy_stats.kurtosis(returns.to_numpy(), bias=False, fisher=True))

    # Handle NaN from scipy
    if math.isnan(skew):
        skew = 0.0
    if math.isnan(kurt):
        kurt = 0.0

    # Standard normal quantile for left tail
    z = scipy_stats.norm.ppf(1 - confidence)

    # Cornish-Fisher expansion
    z2 = z * z
    z3 = z2 * z

    z_cf = (
        z
        + (z2 - 1) * skew / 6
        + (z3 - 3 * z) * kurt / 24
        - (2 * z3 - 5 * z) * skew**2 / 36
    )

    # VaR (negative return at confidence level, expressed as positive loss)
    var = -(mu + z_cf * sigma)

    return float(var)


def entropic_var(
    returns: pl.Series,
    confidence: float = 0.95,
) -> float:
    """Calculate Entropic Value at Risk (EVaR).

    EVaR is a coherent risk measure that provides an upper bound on both
    VaR and CVaR. It is based on the Chernoff inequality and has better
    mathematical properties than VaR.

    Parameters
    ----------
    returns : pl.Series
        Series of returns.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    float
        Entropic VaR (as a positive loss amount).

    Notes
    -----
    EVaR is defined as:
        EVaR_α = inf_{z>0} { (1/z) * ln(E[exp(-z * L)] / (1-α)) }

    Where L is the loss (negative return) and α is the confidence level.

    EVaR satisfies:
        VaR ≤ CVaR ≤ EVaR

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> returns = pl.Series(np.random.normal(0, 0.02, 500))
    >>> evar = entropic_var(returns, confidence=0.95)
    >>> evar > 0
    True

    References
    ----------
    Ahmadi-Javid, A. (2012). "Entropic Value-at-Risk: A New Coherent Risk Measure"
    """
    validate_returns(returns)
    validate_min_length(returns, 10, "entropic_var")

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    returns = to_float_series(returns)
    losses = -returns.to_numpy()  # Convert to losses

    alpha = confidence

    def evar_objective(z: float) -> float:
        """Objective function for EVaR optimization."""
        if z <= 0:
            return float("inf")

        # E[exp(z * L)] for losses L (larger losses increase exponential moment)
        exp_term = np.mean(np.exp(z * losses))

        if exp_term <= 0 or (1 - alpha) <= 0:
            return float("inf")

        return (1 / z) * np.log(exp_term / (1 - alpha))

    # Optimize to find minimum
    try:
        # Search over a reasonable range of z values
        result = optimize.minimize_scalar(
            evar_objective,
            bounds=(0.01, 100),
            method="bounded",
        )

        if result.success:
            return float(result.fun)
        else:
            # Fallback to CVaR if optimization fails
            return _fallback_cvar(returns, confidence)

    except (ValueError, RuntimeWarning, FloatingPointError):
        return _fallback_cvar(returns, confidence)


def _fallback_cvar(returns: pl.Series, confidence: float) -> float:
    """Calculate CVaR as fallback for EVaR."""
    returns_np = returns.to_numpy()
    q = np.percentile(returns_np, (1 - confidence) * 100)
    tail_losses = -returns_np[returns_np <= q]

    if len(tail_losses) == 0:
        return 0.0

    return float(np.mean(tail_losses))


def modified_var(
    returns: pl.Series,
    confidence: float = 0.95,
) -> float:
    """Calculate Modified VaR using Cornish-Fisher expansion.

    This is an alias for cornish_fisher_var with a more intuitive name.

    Parameters
    ----------
    returns : pl.Series
        Series of returns.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    float
        Modified VaR (positive loss amount).
    """
    return cornish_fisher_var(returns, confidence)


def historical_var(
    returns: pl.Series,
    confidence: float = 0.95,
) -> float:
    """Calculate historical (empirical) Value at Risk.

    Uses the empirical distribution of returns to estimate VaR,
    without assuming any parametric distribution.

    Parameters
    ----------
    returns : pl.Series
        Series of returns.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    float
        Historical VaR (positive loss amount).

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> returns = pl.Series(np.random.normal(0, 0.02, 500))
    >>> hvar = historical_var(returns, confidence=0.95)
    >>> hvar > 0
    True
    """
    validate_returns(returns)

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    returns = to_float_series(returns)

    # Quantile at (1 - confidence) level
    q = returns.quantile(1 - confidence, interpolation="linear")

    if q is None:
        return 0.0

    # Return as positive loss
    return float(-q)


def parametric_var(
    returns: pl.Series,
    confidence: float = 0.95,
) -> float:
    """Calculate parametric (Gaussian) Value at Risk.

    Assumes returns are normally distributed and uses the mean and
    standard deviation to calculate VaR.

    Parameters
    ----------
    returns : pl.Series
        Series of returns.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    float
        Parametric VaR (positive loss amount).

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> returns = pl.Series(np.random.normal(0, 0.02, 500))
    >>> pvar = parametric_var(returns, confidence=0.95)
    >>> pvar > 0
    True
    """
    validate_returns(returns)

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    returns = to_float_series(returns)

    mu = returns.mean()
    sigma = returns.std(ddof=1)

    if mu is None or sigma is None or sigma == 0:
        return 0.0

    # Standard normal quantile
    z = scipy_stats.norm.ppf(1 - confidence)

    # VaR
    var = -(float(mu) + z * float(sigma))

    return float(var)
