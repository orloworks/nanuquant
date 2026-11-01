"""Trading metrics for nanuquant.

This module provides algorithmic trading specific metrics that extend
beyond standard QuantStats functionality for systematic trading evaluation.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from nanuquant.config import get_config
from nanuquant.core.distribution import geometric_mean
from nanuquant.core.returns import avg_loss, avg_win, win_rate
from nanuquant.core.utils import (
    safe_divide,
    to_float_series,
)
from nanuquant.core.validation import validate_returns


def exposure(returns: pl.Series) -> float:
    """Calculate market exposure (percent time in market).

    Same definition as QuantStats ``exposure`` (non-zero periods / all periods), but
    exact: QuantStats rounds the result up to two decimals.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Percentage of periods with non-zero returns (0 to 1).

    Notes
    -----
    Returns 1.0 for fully invested strategies (no zero returns).

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.0, -0.02, 0.0, 0.03])
    >>> exposure(returns)
    0.6
    """
    returns = validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    n_periods = len(returns)
    n_active = (returns != 0).sum()

    return float(safe_divide(n_active, n_periods, default=0.0))


def ghpr(returns: pl.Series) -> float:
    """Calculate Geometric Holding Period Return.

    Matches QuantStats ghpr implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Geometric mean of holding period returns.

    Notes
    -----
    GHPR = (1 + total_return)^(1/n) - 1 = geometric mean

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.03])
    >>> ghpr(returns)
    0.0123...
    """
    return geometric_mean(returns)


def rar(
    returns: pl.Series,
    *,
    periods_per_year: int | None = None,
) -> float:
    """Calculate Risk-Adjusted Return (RAR).

    Same definition as QuantStats ``rar`` (CAGR / exposure), using nanuquant's
    period-count CAGR and exact exposure; QuantStats uses calendar-date CAGR and a
    rounded-up exposure.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    periods_per_year : int, optional
        Periods per year for annualization.

    Returns
    -------
    float
        Risk-adjusted return = CAGR / exposure.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.03] * 63)
    >>> rar(returns)
    5.0...
    """
    from nanuquant.core.returns import cagr

    returns = validate_returns(returns)
    if returns.is_empty():
        return 0.0

    config = get_config()
    ppy = periods_per_year or config.periods_per_year

    annual_return = cagr(returns, periods_per_year=ppy)
    exp = exposure(returns)

    if exp == 0:
        return 0.0

    return float(safe_divide(annual_return, exp, default=0.0))


def cpc_index(returns: pl.Series) -> float:
    """Calculate the CPC index.

    Matches QuantStats ``cpc_index``: ``profit_factor * win_rate * payoff_ratio``.
    Values above 1 indicate a robust edge.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        CPC index. ``inf`` if there are wins but no losses.
    """
    from nanuquant.core.returns import payoff_ratio, profit_factor

    returns = validate_returns(returns)
    if returns.is_empty():
        return 0.0
    pf = profit_factor(returns)
    wr = win_rate(returns)
    pr = payoff_ratio(returns)
    if wr == 0:
        return 0.0
    return float(pf * wr * pr)


def serenity_index(returns: pl.Series, *, risk_free_rate: float = 0.0) -> float:
    """Calculate the Serenity index.

    Matches QuantStats ``serenity_index``: ``(sum(r) - rf) / (ulcer_index * pitfall)``,
    where ``pitfall = -cvar(drawdowns) / std(r)`` penalises deep tail drawdowns
    relative to everyday volatility. Higher is better.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    risk_free_rate : float, default 0.0
        Flat amount subtracted from the summed return (QuantStats convention).

    Returns
    -------
    float
        Serenity index. ``inf``/0.0 when there is no drawdown, by the sign of the
        excess return.
    """
    from nanuquant.core.risk import cvar, to_drawdown_series, ulcer_index

    returns = validate_returns(returns)
    if returns.is_empty():
        return 0.0
    returns = to_float_series(returns)
    excess = float(returns.sum()) - risk_free_rate
    std = returns.std()
    ui = ulcer_index(returns)
    if std is None or std == 0 or ui == 0:
        return float("inf") if excess > 0 else 0.0
    pitfall = -cvar(to_drawdown_series(returns)) / float(std)
    if pitfall == 0:
        return float("inf") if excess > 0 else 0.0
    return float(excess / (ui * pitfall))


def risk_of_ruin(returns: pl.Series) -> float:
    """Calculate the QuantStats risk-of-ruin heuristic.

    Matches QuantStats ``risk_of_ruin``: ``((1 - W) / (1 + W)) ** n`` with ``W`` the
    :func:`win_rate` and ``n`` the number of periods. This is a coarse heuristic that
    goes to 0 for any long series with a win rate above 0; it is computed in log space
    so it cannot overflow.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Probability-like value in [0, 1].
    """
    returns = validate_returns(returns)
    if returns.is_empty():
        return 1.0
    w = win_rate(returns)
    if w <= 0:
        return 1.0
    log_ratio = math.log((1 - w) / (1 + w)) if w < 1 else float("-inf")
    return float(math.exp(len(returns) * log_ratio))


def adjusted_sortino(
    returns: pl.Series,
    *,
    risk_free_rate: float | None = None,
    periods_per_year: int | None = None,
) -> float:
    """Calculate the Sortino ratio divided by sqrt(2).

    Matches QuantStats ``adjusted_sortino`` (Jack Schwager's adjustment): dividing by
    ``sqrt(2)`` makes the Sortino ratio directly comparable with the Sharpe ratio.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    risk_free_rate : float, optional
        Annualized risk-free rate.
    periods_per_year : int, optional
        Periods per year for annualization.

    Returns
    -------
    float
        ``sortino / sqrt(2)``.
    """
    from nanuquant.core.performance import sortino

    returns = validate_returns(returns)
    if returns.is_empty():
        return 0.0
    return float(
        sortino(returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
        / math.sqrt(2)
    )


def smart_sharpe(
    returns: pl.Series,
    *,
    risk_free_rate: float | None = None,
    periods_per_year: int | None = None,
) -> float:
    """Calculate Sharpe ratio adjusted for autocorrelation.

    Matches QuantStats smart_sharpe implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    risk_free_rate : float, optional
        Annualized risk-free rate.
    periods_per_year : int, optional
        Periods per year for annualization.

    Returns
    -------
    float
        Smart Sharpe = Sharpe / autocorr_penalty (QuantStats)

    Notes
    -----
    Adjusts for autocorrelation which can inflate Sharpe ratio.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.03] * 50)
    >>> smart_sharpe(returns)
    1.5...
    """
    from nanuquant.core.performance import sharpe

    returns = validate_returns(returns)
    if returns.is_empty() or len(returns) < 3:
        return 0.0

    config = get_config()
    rf = risk_free_rate if risk_free_rate is not None else config.risk_free_rate
    ppy = periods_per_year or config.periods_per_year

    sharpe_ratio = sharpe(returns, risk_free_rate=rf, periods_per_year=ppy)

    returns = to_float_series(returns)
    return float(sharpe_ratio / _autocorr_penalty(returns))


def smart_sortino(
    returns: pl.Series,
    *,
    risk_free_rate: float | None = None,
    periods_per_year: int | None = None,
) -> float:
    """Calculate Sortino ratio adjusted for autocorrelation.

    Matches QuantStats smart_sortino implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    risk_free_rate : float, optional
        Annualized risk-free rate.
    periods_per_year : int, optional
        Periods per year for annualization.

    Returns
    -------
    float
        Smart Sortino = Sortino / autocorr_penalty (QuantStats)

    Notes
    -----
    Adjusts for autocorrelation which can inflate Sortino ratio.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.03] * 50)
    >>> smart_sortino(returns)
    2.0...
    """
    from nanuquant.core.performance import sortino

    returns = validate_returns(returns)
    if returns.is_empty() or len(returns) < 3:
        return 0.0

    config = get_config()
    rf = risk_free_rate if risk_free_rate is not None else config.risk_free_rate
    ppy = periods_per_year or config.periods_per_year

    sortino_ratio = sortino(returns, risk_free_rate=rf, periods_per_year=ppy)

    returns = to_float_series(returns)
    return float(sortino_ratio / _autocorr_penalty(returns))


def _autocorr_penalty(returns: pl.Series) -> float:
    """QuantStats autocorrelation penalty.

    ``sqrt(1 + 2 * sum_{k=1}^{n-1} ((n - k) / n) * |rho| ** k)``, with ``rho`` the
    lag-1 Pearson correlation of the series with itself.
    """
    arr = returns.to_numpy()
    n = len(arr)
    if n < 3:
        return 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        coef = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
    if math.isnan(coef):
        return 1.0
    coef = abs(coef)
    k = np.arange(1, n, dtype=float)
    weights = (n - k) / n
    # coef**k underflows harmlessly to 0 for large k
    return float(math.sqrt(1 + 2 * float(np.sum(weights * coef**k))))


def sqn(returns: pl.Series) -> float:
    """Calculate System Quality Number (SQN).

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        SQN = sqrt(n) * mean(returns) / std(returns)

    Notes
    -----
    Van Tharp's SQN measures system quality:
    - 1.6-1.9: Below average
    - 2.0-2.4: Average
    - 2.5-2.9: Good
    - 3.0-5.0: Excellent
    - 5.1+: Superb (rare)

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, -0.01, 0.03, -0.01, 0.01] * 50)
    >>> sqn(returns)
    2.5...
    """
    returns = validate_returns(returns)
    if returns.is_empty() or len(returns) < 2:
        return 0.0

    returns = to_float_series(returns)
    n = len(returns)

    mean_ret = returns.mean()
    std_ret = returns.std()

    if mean_ret is None or std_ret is None or std_ret == 0:
        return 0.0

    return float(math.sqrt(n) * mean_ret / std_ret)


def expectancy(returns: pl.Series) -> float:
    """Calculate trade expectancy.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    Notes
    -----
    Expected value per trade. Positive expectancy required for profitability.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, -0.01, 0.03, -0.01, 0.01])
    >>> expectancy(returns)
    0.008...
    """
    returns = validate_returns(returns)
    if returns.is_empty():
        return 0.0

    wr = win_rate(returns)
    avg_w = avg_win(returns)
    avg_l = avg_loss(returns)

    return float(wr * avg_w + (1 - wr) * avg_l)


def k_ratio(returns: pl.Series) -> float:
    """Calculate K-Ratio (Lars Kestner).

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        K-ratio = slope of equity curve / standard error of slope.

    Notes
    -----
    Measures consistency of equity curve growth.
    Higher values indicate more consistent upward slope.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.03] * 50)
    >>> k_ratio(returns)
    0.5...
    """
    returns = validate_returns(returns)
    if returns.is_empty() or len(returns) < 3:
        return 0.0

    returns = to_float_series(returns)
    n = len(returns)

    # Create cumulative log returns (equity curve)
    cum_returns = returns.cum_sum().to_numpy()
    x = list(range(n))

    # Linear regression
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, cum_returns)

    if std_err == 0:
        return float("inf") if slope > 0 else 0.0

    return float(slope / std_err)
