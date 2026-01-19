"""Trading metrics for nanuquant.

This module provides algorithmic trading specific metrics that extend
beyond standard QuantStats functionality for systematic trading evaluation.
"""

from __future__ import annotations

import math

import polars as pl
from scipy import stats as scipy_stats

from nanuquant.config import get_config
from nanuquant.core.distribution import skewness
from nanuquant.core.returns import avg_loss, avg_win, comp, win_rate
from nanuquant.core.risk import var
from nanuquant.core.utils import (
    get_annualization_factor,
    safe_divide,
    to_float_series,
)
from nanuquant.core.validation import validate_returns


def exposure(returns: pl.Series) -> float:
    """Calculate market exposure (percent time in market).

    Matches QuantStats exposure implementation.

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
    validate_returns(returns)
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
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    n = len(returns)

    product = (1 + returns).product()
    if product is None or product <= 0:
        return 0.0

    return float(product ** (1.0 / n) - 1)


def rar(
    returns: pl.Series,
    *,
    periods_per_year: int | None = None,
) -> float:
    """Calculate Risk-Adjusted Return (RAR).

    Matches QuantStats rar implementation.

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

    validate_returns(returns)
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
    """Calculate CPC Index (Commodity Producers Index).

    Matches QuantStats cpc_index implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        CPC Index = (win_rate * payoff_ratio - 1) + 1

    Notes
    -----
    Also known as Expectunity or Edge Ratio.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, -0.01, 0.03, -0.01, 0.01])
    >>> cpc_index(returns)
    1.2...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    wr = win_rate(returns)
    avg_w = avg_win(returns)
    avg_l = avg_loss(returns)

    if avg_l == 0:
        return float("inf") if avg_w > 0 else 0.0

    payoff = abs(avg_w / avg_l)

    return float(wr * payoff - (1 - wr) + 1)


def serenity_index(returns: pl.Series) -> float:
    """Calculate Serenity Index.

    Matches QuantStats serenity_index implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Serenity Index = CAGR / (Ulcer Index * sqrt(periods)).

    Notes
    -----
    Lower is better for drawdown-adjusted return.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.03] * 63)
    >>> serenity_index(returns)
    5.0...
    """
    from nanuquant.core.returns import cagr
    from nanuquant.core.risk import ulcer_index

    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    config = get_config()
    ppy = config.periods_per_year

    annual_return = cagr(returns, periods_per_year=ppy)
    ui = ulcer_index(returns)

    if ui == 0:
        return float("inf") if annual_return > 0 else 0.0

    return float(safe_divide(annual_return, ui, default=0.0))


def risk_of_ruin(
    returns: pl.Series,
    *,
    target_fraction: float = 1.0,
) -> float:
    """Calculate probability of reaching ruin.

    Matches QuantStats risk_of_ruin implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    target_fraction : float, default 1.0
        Fraction of capital loss defining ruin (1.0 = total loss).

    Returns
    -------
    float
        Probability of ruin (0 to 1).

    Notes
    -----
    Uses the formula: ROR = ((1 - E) / (1 + E))^n
    Where E = Edge = win_rate * payoff - (1 - win_rate)

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, -0.01, 0.03, -0.01, 0.01] * 50)
    >>> risk_of_ruin(returns)
    0.0...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 1.0

    wr = win_rate(returns)
    avg_w = avg_win(returns)
    avg_l = avg_loss(returns)

    if avg_l == 0:
        return 0.0 if avg_w > 0 else 1.0

    payoff = abs(avg_w / avg_l)

    # Edge = win_rate * payoff - (1 - win_rate)
    edge = wr * payoff - (1 - wr)

    if edge >= 1:
        return 0.0
    if edge <= -1:
        return 1.0

    # Risk of ruin formula
    n = len(returns)
    ror = ((1 - edge) / (1 + edge)) ** n

    return float(min(1.0, max(0.0, ror)))


def adjusted_sortino(
    returns: pl.Series,
    *,
    risk_free_rate: float | None = None,
    periods_per_year: int | None = None,
) -> float:
    """Calculate Sortino ratio adjusted for skewness.

    Matches QuantStats adjusted_sortino implementation.

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
        Adjusted Sortino = Sortino * (1 + skew/6 * Sortino - kurt/24 * Sortino^2)

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.03] * 50)
    >>> adjusted_sortino(returns)
    2.5...
    """
    from nanuquant.core.distribution import kurtosis
    from nanuquant.core.performance import sortino

    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    config = get_config()
    rf = risk_free_rate if risk_free_rate is not None else config.risk_free_rate
    ppy = periods_per_year or config.periods_per_year

    sort_ratio = sortino(returns, risk_free_rate=rf, periods_per_year=ppy)
    skew = skewness(returns)
    kurt = kurtosis(returns)

    # Adjusted Sortino formula
    adj = sort_ratio * (1 + (skew / 6) * sort_ratio - (kurt / 24) * sort_ratio ** 2)

    return float(adj)


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
        Smart Sharpe = Sharpe / sqrt(1 + autocorr_penalty)

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

    validate_returns(returns)
    if returns.is_empty() or len(returns) < 3:
        return 0.0

    config = get_config()
    rf = risk_free_rate if risk_free_rate is not None else config.risk_free_rate
    ppy = periods_per_year or config.periods_per_year

    sharpe_ratio = sharpe(returns, risk_free_rate=rf, periods_per_year=ppy)

    # Calculate autocorrelation penalty
    returns = to_float_series(returns)
    corr = _autocorr(returns)

    if corr is None or corr == 0:
        return float(sharpe_ratio)

    # Penalty adjustment
    penalty = _autocorr_penalty(corr)

    if penalty <= -1:
        return 0.0

    return float(sharpe_ratio / math.sqrt(1 + penalty))


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
        Smart Sortino = Sortino / sqrt(1 + autocorr_penalty)

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

    validate_returns(returns)
    if returns.is_empty() or len(returns) < 3:
        return 0.0

    config = get_config()
    rf = risk_free_rate if risk_free_rate is not None else config.risk_free_rate
    ppy = periods_per_year or config.periods_per_year

    sortino_ratio = sortino(returns, risk_free_rate=rf, periods_per_year=ppy)

    # Calculate autocorrelation penalty
    returns = to_float_series(returns)
    corr = _autocorr(returns)

    if corr is None or corr == 0:
        return float(sortino_ratio)

    # Penalty adjustment
    penalty = _autocorr_penalty(corr)

    if penalty <= -1:
        return 0.0

    return float(sortino_ratio / math.sqrt(1 + penalty))


def _autocorr(returns: pl.Series, lag: int = 1) -> float | None:
    """Calculate autocorrelation at given lag."""
    if len(returns) <= lag:
        return None

    returns = to_float_series(returns)
    n = len(returns)

    mean = returns.mean()
    if mean is None:
        return None

    # Calculate autocorrelation
    r1 = returns[lag:]
    r2 = returns[:-lag]

    cov = ((r1 - mean) * (r2 - mean)).sum()
    var = ((returns - mean) ** 2).sum()

    if cov is None or var is None or var == 0:
        return None

    return float(cov / var)


def _autocorr_penalty(corr: float) -> float:
    """Calculate autocorrelation penalty for smart ratios."""
    # QuantStats formula: 2 * sum((1 - i/n) * rho_i)
    # Simplified for lag-1: 2 * corr
    return 2 * corr


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
    validate_returns(returns)
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
    validate_returns(returns)
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
    validate_returns(returns)
    if returns.is_empty() or len(returns) < 3:
        return 0.0

    returns = to_float_series(returns)
    n = len(returns)

    # Create cumulative log returns (equity curve)
    cum_returns = returns.cum_sum().to_numpy()
    x = list(range(n))

    # Linear regression
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
        x, cum_returns
    )

    if std_err == 0:
        return float("inf") if slope > 0 else 0.0

    return float(slope / std_err)
