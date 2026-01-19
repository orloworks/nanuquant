"""Volatility modeling metrics for regime detection.

This module provides conditional volatility models and tests for volatility
clustering, which are essential for understanding market regimes and
time-varying risk.

References
----------
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"
- Engle, R. F. (1982). "Autoregressive Conditional Heteroskedasticity with
  Estimates of the Variance of United Kingdom Inflation"
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from nanuquant.core.utils import to_float_series
from nanuquant.core.validation import validate_min_length, validate_returns


class ARCHTestResult(NamedTuple):
    """Result of ARCH effect test.

    Attributes
    ----------
    statistic : float
        The LM test statistic (n * R^2).
    p_value : float
        P-value for the null hypothesis of no ARCH effects.
    lags : int
        Number of lags used in the test.
    has_arch_effects : bool
        True if null hypothesis rejected at 5% level (p < 0.05).
    """

    statistic: float
    p_value: float
    lags: int
    has_arch_effects: bool


class GARCHResult(NamedTuple):
    """Result of GARCH(1,1) volatility estimation.

    Attributes
    ----------
    omega : float
        Constant term in the GARCH equation.
    alpha : float
        ARCH coefficient (weight on lagged squared returns).
    beta : float
        GARCH coefficient (weight on lagged conditional variance).
    conditional_volatility : pl.Series
        Time series of conditional volatility estimates.
    long_run_variance : float
        Unconditional (long-run) variance: omega / (1 - alpha - beta).
    persistence : float
        Volatility persistence: alpha + beta. Values close to 1 indicate
        highly persistent volatility shocks.
    forecast : float
        One-step-ahead volatility forecast.
    """

    omega: float
    alpha: float
    beta: float
    conditional_volatility: pl.Series
    long_run_variance: float
    persistence: float
    forecast: float


def arch_effect_test(
    returns: pl.Series,
    lags: int = 12,
) -> ARCHTestResult:
    """Test for ARCH effects using Engle's Lagrange Multiplier test.

    This test checks for the presence of volatility clustering (ARCH effects)
    in the return series by regressing squared residuals on their lagged values.

    Parameters
    ----------
    returns : pl.Series
        Series of returns to test.
    lags : int, default 12
        Number of lags to include in the test. More lags increase power
        for detecting long-memory ARCH effects but reduce degrees of freedom.

    Returns
    -------
    ARCHTestResult
        Named tuple containing:
        - statistic: LM test statistic (n * R^2)
        - p_value: P-value from chi-squared distribution
        - lags: Number of lags used
        - has_arch_effects: True if p_value < 0.05

    Notes
    -----
    The null hypothesis is that there are no ARCH effects (homoskedasticity).
    A low p-value (< 0.05) indicates significant volatility clustering.

    The test statistic follows a chi-squared distribution with `lags` degrees
    of freedom under the null hypothesis.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> returns = pl.Series(np.random.normal(0, 0.02, 500))
    >>> result = arch_effect_test(returns)
    >>> result.has_arch_effects
    False

    References
    ----------
    Engle, R. F. (1982). "Autoregressive Conditional Heteroskedasticity with
    Estimates of the Variance of United Kingdom Inflation"
    """
    validate_returns(returns)
    validate_min_length(returns, lags + 2, "arch_effect_test")

    returns = to_float_series(returns)
    n = len(returns)

    # Demean returns
    mean = returns.mean()
    if mean is None:
        return ARCHTestResult(
            statistic=0.0, p_value=1.0, lags=lags, has_arch_effects=False
        )

    residuals = (returns - mean).to_numpy()

    # Squared residuals
    squared_residuals = residuals**2

    # Prepare regression: squared_residuals[lags:] ~ const + lagged_squared[:-lags]
    y = squared_residuals[lags:]
    n_obs = len(y)

    # Create lagged squared residuals matrix
    X = np.zeros((n_obs, lags + 1))
    X[:, 0] = 1  # Constant term
    for i in range(1, lags + 1):
        X[:, i] = squared_residuals[lags - i : n - i]

    # OLS regression
    try:
        # Use numpy's least squares solver
        beta, residual_ss, rank, s = np.linalg.lstsq(X, y, rcond=None)

        # Predicted values and R-squared
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            r_squared = 0.0
        else:
            r_squared = 1 - (ss_res / ss_tot)
            r_squared = max(0.0, r_squared)  # Ensure non-negative

        # LM test statistic
        lm_stat = n_obs * r_squared

        # P-value from chi-squared distribution
        p_value = 1 - scipy_stats.chi2.cdf(lm_stat, df=lags)

        return ARCHTestResult(
            statistic=float(lm_stat),
            p_value=float(p_value),
            lags=lags,
            has_arch_effects=p_value < 0.05,
        )

    except np.linalg.LinAlgError:
        return ARCHTestResult(
            statistic=0.0, p_value=1.0, lags=lags, has_arch_effects=False
        )


def garch_volatility(
    returns: pl.Series,
    p: int = 1,
    q: int = 1,
    *,
    forecast_horizon: int = 1,
) -> GARCHResult:
    """Estimate GARCH(p,q) conditional volatility.

    Estimates time-varying volatility using a GARCH(1,1) model. Will use the
    `arch` library if available for maximum likelihood estimation, otherwise
    falls back to a variance targeting method.

    Parameters
    ----------
    returns : pl.Series
        Series of returns.
    p : int, default 1
        Order of GARCH term (lagged conditional variance). Only p=1 is
        currently supported.
    q : int, default 1
        Order of ARCH term (lagged squared returns). Only q=1 is
        currently supported.
    forecast_horizon : int, default 1
        Number of periods ahead to forecast volatility.

    Returns
    -------
    GARCHResult
        Named tuple containing:
        - omega: Constant term
        - alpha: ARCH coefficient
        - beta: GARCH coefficient
        - conditional_volatility: Series of volatility estimates
        - long_run_variance: Unconditional variance
        - persistence: alpha + beta
        - forecast: Volatility forecast

    Notes
    -----
    GARCH(1,1) model:
        σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}

    Where:
        - ω > 0, α ≥ 0, β ≥ 0
        - α + β < 1 for stationarity

    The fallback implementation uses variance targeting:
        ω = σ² * (1 - α - β)

    Where σ² is the unconditional variance and typical starting values
    α = 0.1, β = 0.85 are used.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> returns = pl.Series(np.random.normal(0, 0.02, 500))
    >>> result = garch_volatility(returns)
    >>> result.persistence < 1
    True

    References
    ----------
    Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"
    """
    validate_returns(returns)
    validate_min_length(returns, 20, "garch_volatility")

    if p != 1 or q != 1:
        raise ValueError("Only GARCH(1,1) is currently supported (p=1, q=1)")

    returns = to_float_series(returns)

    # Try to use arch library if available
    try:
        return _garch_arch_library(returns, forecast_horizon)
    except ImportError:
        pass

    # Fallback to variance targeting method
    return _garch_variance_targeting(returns, forecast_horizon)


def _garch_arch_library(
    returns: pl.Series,
    forecast_horizon: int = 1,
) -> GARCHResult:
    """Estimate GARCH using the arch library (MLE)."""
    from arch import arch_model

    returns_np = returns.to_numpy() * 100  # arch library expects percentage returns

    model = arch_model(returns_np, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
    result = model.fit(disp="off", show_warning=False)

    # Extract parameters (scale back from percentage)
    omega = result.params["omega"] / 10000  # Scale back from pct^2
    alpha = result.params["alpha[1]"]
    beta = result.params["beta[1]"]

    # Conditional volatility (scale back from percentage)
    cond_vol_pct = result.conditional_volatility
    cond_vol = pl.Series("conditional_volatility", cond_vol_pct / 100)

    # Long-run variance
    persistence = alpha + beta
    if persistence >= 1:
        long_run_var = cond_vol.to_numpy()[-1] ** 2
    else:
        long_run_var = omega / (1 - persistence)

    # Forecast
    forecast_result = result.forecast(horizon=forecast_horizon)
    forecast_var = forecast_result.variance.values[-1, -1] / 10000
    forecast_vol = math.sqrt(forecast_var)

    return GARCHResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        conditional_volatility=cond_vol,
        long_run_variance=long_run_var,
        persistence=persistence,
        forecast=forecast_vol,
    )


def _garch_variance_targeting(
    returns: pl.Series,
    forecast_horizon: int = 1,
) -> GARCHResult:
    """Estimate GARCH using variance targeting (fallback method).

    Uses fixed starting values and variance targeting to estimate
    GARCH(1,1) parameters without MLE optimization.
    """
    returns_np = returns.to_numpy()
    n = len(returns_np)

    # Sample variance for variance targeting
    sample_var = np.var(returns_np, ddof=1)

    # Typical GARCH(1,1) parameters
    # These are reasonable defaults based on empirical research
    alpha = 0.10  # ARCH coefficient
    beta = 0.85  # GARCH coefficient
    persistence = alpha + beta

    # Variance targeting: omega = sample_var * (1 - alpha - beta)
    omega = sample_var * (1 - persistence)

    # Initialize conditional variance
    cond_var = np.zeros(n)
    cond_var[0] = sample_var

    # GARCH recursion
    for t in range(1, n):
        cond_var[t] = omega + alpha * returns_np[t - 1] ** 2 + beta * cond_var[t - 1]

    # Conditional volatility
    cond_vol = pl.Series("conditional_volatility", np.sqrt(cond_var))

    # Long-run variance
    long_run_var = omega / (1 - persistence)

    # One-step ahead forecast
    last_return = returns_np[-1]
    last_var = cond_var[-1]
    forecast_var = omega + alpha * last_return**2 + beta * last_var

    # Multi-step forecast (if needed)
    for h in range(1, forecast_horizon):
        forecast_var = omega + persistence * forecast_var

    forecast_vol = math.sqrt(forecast_var)

    return GARCHResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        conditional_volatility=cond_vol,
        long_run_variance=long_run_var,
        persistence=persistence,
        forecast=forecast_vol,
    )
