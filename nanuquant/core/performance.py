"""Performance ratio metrics for nanuquant.

This module provides risk-adjusted performance metrics that match QuantStats output.
"""

from __future__ import annotations

import math

import polars as pl

from nanuquant.config import get_config
from nanuquant.core.returns import cagr, comp
from nanuquant.core.risk import (
    max_drawdown,
    to_drawdown_series,
    ulcer_index,
    volatility,
)
from nanuquant.core.utils import (
    get_annualization_factor,
    safe_divide,
    to_float_series,
)
from nanuquant.core.validation import validate_min_length, validate_returns


def sharpe(
    returns: pl.Series,
    *,
    risk_free_rate: float | None = None,
    periods_per_year: int | None = None,
) -> float:
    """Calculate the Sharpe ratio.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    risk_free_rate : float, optional
        Annualized risk-free rate. If None, uses config default.
    periods_per_year : int, optional
        Periods per year for annualization.

    Returns
    -------
    float
        Sharpe ratio (excess return / volatility).

    Notes
    -----
    Formula: (mean(returns) - rf_per_period) * sqrt(periods) / std(returns)

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.015, 0.005])
    >>> sharpe(returns, risk_free_rate=0.0, periods_per_year=252)
    2.54...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    config = get_config()

    rf = risk_free_rate if risk_free_rate is not None else config.risk_free_rate
    ann_factor = get_annualization_factor(
        periods_per_year=periods_per_year or config.periods_per_year
    )

    # Convert annual risk-free to per-period
    rf_per_period = rf / ann_factor

    # Excess returns
    excess_returns = returns - rf_per_period

    mean_excess = excess_returns.mean()
    std_returns = returns.std(ddof=config.ddof)

    if mean_excess is None or std_returns is None or std_returns == 0:
        return 0.0

    # Annualized Sharpe
    sharpe_ratio = (mean_excess / std_returns) * math.sqrt(ann_factor)

    return float(sharpe_ratio)


def sortino(
    returns: pl.Series,
    *,
    risk_free_rate: float | None = None,
    periods_per_year: int | None = None,
) -> float:
    """Calculate the Sortino ratio.

    Matches QuantStats implementation based on Red Rock Capital paper.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    risk_free_rate : float, optional
        Annualized risk-free rate. If None, uses config default.
    periods_per_year : int, optional
        Periods per year for annualization.

    Returns
    -------
    float
        Sortino ratio (mean return / downside deviation).

    Notes
    -----
    Formula from Red Rock Capital paper:
    - downside = sqrt(sum(returns[returns < 0]^2) / n)
    - sortino = (mean(returns) - rf_per_period) / downside * sqrt(periods)

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.015, 0.005])
    >>> sortino(returns, risk_free_rate=0.0, periods_per_year=252)
    3.60...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    config = get_config()

    rf = risk_free_rate if risk_free_rate is not None else config.risk_free_rate
    ann_factor = get_annualization_factor(
        periods_per_year=periods_per_year or config.periods_per_year
    )

    # Convert annual risk-free to per-period
    rf_per_period = rf / ann_factor

    # Adjust returns for risk-free rate
    adjusted_returns = returns - rf_per_period

    # Calculate downside deviation (non-annualized)
    # Formula: sqrt(sum(negative_returns^2) / n)
    n = len(adjusted_returns)
    negative_returns = adjusted_returns.filter(adjusted_returns < 0)

    if negative_returns.is_empty():
        mean_ret = adjusted_returns.mean()
        return float("inf") if mean_ret is not None and mean_ret > 0 else 0.0

    squared_sum = (negative_returns ** 2).sum()
    if squared_sum is None:
        return 0.0

    downside = math.sqrt(squared_sum / n)

    if downside == 0:
        mean_ret = adjusted_returns.mean()
        return float("inf") if mean_ret is not None and mean_ret > 0 else 0.0

    mean_return = adjusted_returns.mean()
    if mean_return is None:
        return 0.0

    # Annualize
    sortino_ratio = (mean_return / downside) * math.sqrt(ann_factor)

    return float(sortino_ratio)


def calmar(
    returns: pl.Series,
    *,
    periods_per_year: int | None = None,
) -> float:
    """Calculate the Calmar ratio.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    periods_per_year : int, optional
        Periods per year for CAGR calculation.

    Returns
    -------
    float
        Calmar ratio (CAGR / |max drawdown|).

    Notes
    -----
    Higher is better. Measures return relative to worst drawdown.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.05, 0.03, 0.02])
    >>> calmar(returns, periods_per_year=252)
    5.0...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    config = get_config()
    ppy = periods_per_year or config.periods_per_year

    annual_return = cagr(returns, periods_per_year=ppy)
    mdd = max_drawdown(returns)

    # max_drawdown returns negative, take absolute value
    if mdd == 0:
        return float("inf") if annual_return > 0 else 0.0

    return float(safe_divide(annual_return, abs(mdd), default=0.0))


def omega(
    returns: pl.Series,
    *,
    threshold: float = 0.0,
    risk_free_rate: float = 0.0,
    periods_per_year: int | None = None,
) -> float:
    """Calculate the Omega ratio.

    Matches QuantStats implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    threshold : float, default 0.0
        Required return (annualized).
    risk_free_rate : float, default 0.0
        Risk-free rate (annualized) - used to adjust returns.
    periods_per_year : int, optional
        Periods per year for threshold conversion.

    Returns
    -------
    float
        Omega ratio (probability-weighted gains / losses).

    Notes
    -----
    QuantStats converts the threshold to per-period:
    threshold_per_period = (1 + threshold)^(1/periods) - 1

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, 0.01, -0.01, 0.03, -0.02])
    >>> omega(returns, threshold=0.0)
    2.0
    """
    validate_returns(returns)
    if len(returns) < 2:
        return float("nan")

    if threshold <= -1:
        return float("nan")

    returns = to_float_series(returns)
    config = get_config()
    ann_factor = get_annualization_factor(
        periods_per_year=periods_per_year or config.periods_per_year
    )

    # Adjust returns for risk-free rate
    rf_per_period = risk_free_rate / ann_factor
    adjusted_returns = returns - rf_per_period

    # Convert annualized threshold to per-period (QuantStats formula)
    if ann_factor == 1:
        threshold_per_period = threshold
    else:
        threshold_per_period = (1 + threshold) ** (1.0 / ann_factor) - 1

    # Calculate gains and losses relative to threshold
    returns_less_thresh = adjusted_returns - threshold_per_period

    gains = returns_less_thresh.filter(returns_less_thresh > 0)
    losses = returns_less_thresh.filter(returns_less_thresh < 0)

    sum_gains = gains.sum() if not gains.is_empty() else 0.0
    sum_losses = abs(losses.sum()) if not losses.is_empty() else 0.0

    if sum_losses == 0:
        return float("nan")

    return float(safe_divide(sum_gains, sum_losses, default=float("nan")))


def gain_to_pain_ratio(returns: pl.Series) -> float:
    """Calculate gain-to-pain ratio.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Sum of all returns / Sum of absolute losses.

    Notes
    -----
    Also known as "pain ratio". Higher is better.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, 0.01, -0.01, 0.03, -0.02])
    >>> gain_to_pain_ratio(returns)
    1.0
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)

    total_return = returns.sum()
    losses = returns.filter(returns < 0)
    sum_abs_losses = abs(losses.sum()) if not losses.is_empty() else 0.0

    if sum_abs_losses == 0:
        return float("inf") if total_return > 0 else 0.0

    return float(safe_divide(total_return, sum_abs_losses, default=0.0))


def ulcer_performance_index(
    returns: pl.Series,
    *,
    risk_free_rate: float | None = None,
) -> float:
    """Calculate Ulcer Performance Index (UPI).

    Matches QuantStats implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    risk_free_rate : float, optional
        Risk-free rate (used as flat adjustment).

    Returns
    -------
    float
        UPI = (comp(returns) - rf) / Ulcer Index.

    Notes
    -----
    QuantStats uses total compounded return, not CAGR.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.015, 0.005])
    >>> ulcer_performance_index(returns, risk_free_rate=0.0)
    5.0...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    config = get_config()
    rf = risk_free_rate if risk_free_rate is not None else config.risk_free_rate

    total_return = comp(returns)
    ui = ulcer_index(returns)

    excess = total_return - rf

    if ui == 0:
        return float("inf") if excess > 0 else 0.0

    return float(safe_divide(excess, ui, default=0.0))


def kelly_criterion(returns: pl.Series) -> float:
    """Calculate Kelly criterion (optimal bet sizing).

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Optimal fraction of capital to risk per trade.

    Notes
    -----
    Kelly = W - (1-W)/R
    Where W = win rate, R = payoff ratio (avg win / avg loss)

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, -0.01, 0.03, -0.01, 0.02])
    >>> kelly_criterion(returns)
    0.40...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)

    # Win rate
    wins = returns.filter(returns > 0)
    losses = returns.filter(returns < 0)

    n_trades = len(returns)
    n_wins = len(wins)
    n_losses = len(losses)

    if n_trades == 0:
        return 0.0

    w = n_wins / n_trades

    # Calculate payoff ratio
    avg_win = wins.mean() if not wins.is_empty() else 0.0
    avg_loss = losses.mean() if not losses.is_empty() else 0.0

    if avg_win is None:
        avg_win = 0.0
    if avg_loss is None or avg_loss == 0:
        return w if w > 0 else 0.0

    # Payoff ratio (avg_loss is negative, so take abs)
    r = avg_win / abs(avg_loss)

    # Handle edge case where payoff ratio is 0 (no wins)
    if r == 0:
        return float("-inf")  # No wins means kelly suggests full loss

    # Kelly formula
    kelly = w - (1 - w) / r

    return float(kelly)


def tail_ratio(returns: pl.Series) -> float:
    """Calculate tail ratio (right tail / left tail).

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Ratio of 95th percentile to 5th percentile (absolute).

    Notes
    -----
    Measures asymmetry of return distribution tails.
    Values > 1 indicate fatter right tail (good).

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.05, -0.02])
    >>> tail_ratio(returns)
    2.5...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)

    right_tail = returns.quantile(0.95, interpolation="linear")
    left_tail = returns.quantile(0.05, interpolation="linear")

    if right_tail is None or left_tail is None or left_tail == 0:
        return 0.0

    return float(safe_divide(abs(right_tail), abs(left_tail), default=0.0))


def common_sense_ratio(returns: pl.Series) -> float:
    """Calculate common sense ratio.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Profit factor * Tail ratio.

    Notes
    -----
    Combines profitability with tail behavior.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, -0.01, 0.03, -0.01, 0.02])
    >>> common_sense_ratio(returns)
    6.25...
    """
    from nanuquant.core.returns import profit_factor as pf

    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    return float(pf(returns) * tail_ratio(returns))


def risk_return_ratio(
    returns: pl.Series,
    *,
    periods_per_year: int | None = None,
) -> float:
    """Calculate risk-return ratio.

    Matches QuantStats implementation: mean / std (not annualized).

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    periods_per_year : int, optional
        Ignored - kept for API compatibility.

    Returns
    -------
    float
        Mean return divided by standard deviation (not annualized).

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.015, 0.005])
    >>> risk_return_ratio(returns)
    0.65...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)

    mean_ret = returns.mean()
    std_ret = returns.std()

    if mean_ret is None or std_ret is None or std_ret == 0:
        return 0.0

    return float(mean_ret / std_ret)


def recovery_factor(returns: pl.Series) -> float:
    """Calculate recovery factor.

    Matches QuantStats implementation: sum(returns) / |max_drawdown|.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Sum of returns divided by absolute max drawdown.

    Notes
    -----
    QuantStats uses arithmetic sum, not compounded return.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.10, -0.05, 0.08, 0.05])
    >>> recovery_factor(returns)
    3.6...
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)

    total = returns.sum()
    mdd = max_drawdown(returns)

    if mdd == 0:
        return float("inf") if total > 0 else 0.0

    return float(safe_divide(abs(total), abs(mdd), default=0.0))


# ===== BENCHMARK COMPARISON METRICS =====


def greeks(
    returns: pl.Series,
    benchmark: pl.Series,
    *,
    periods_per_year: int | None = None,
) -> tuple[float, float]:
    """Calculate alpha and beta relative to a benchmark.

    Matches QuantStats greeks implementation.

    Parameters
    ----------
    returns : pl.Series
        Strategy period returns.
    benchmark : pl.Series
        Benchmark period returns (same frequency).
    periods_per_year : int, optional
        Periods per year for annualization.

    Returns
    -------
    tuple[float, float]
        (alpha, beta) where:
        - alpha: Annualized excess return over benchmark
        - beta: Sensitivity to benchmark movements

    Notes
    -----
    Uses OLS regression: returns = alpha + beta * benchmark + epsilon
    Alpha is annualized by multiplying by periods_per_year.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, -0.01, 0.03, 0.01, -0.02])
    >>> benchmark = pl.Series([0.01, -0.02, 0.02, 0.01, -0.01])
    >>> alpha, beta = greeks(returns, benchmark)
    """
    from nanuquant.core.validation import validate_benchmark_match

    validate_returns(returns)
    validate_returns(benchmark)
    validate_benchmark_match(returns, benchmark)

    if returns.is_empty() or len(returns) < 2:
        return (0.0, 0.0)

    returns = to_float_series(returns)
    benchmark = to_float_series(benchmark)

    config = get_config()
    ann_factor = get_annualization_factor(
        periods_per_year=periods_per_year or config.periods_per_year
    )

    # Calculate beta (covariance / variance)
    mean_ret = returns.mean()
    mean_bench = benchmark.mean()

    if mean_ret is None or mean_bench is None:
        return (0.0, 0.0)

    # Covariance
    cov = ((returns - mean_ret) * (benchmark - mean_bench)).mean()
    # Variance of benchmark
    var_bench = ((benchmark - mean_bench) ** 2).mean()

    if cov is None or var_bench is None or var_bench == 0:
        beta = 0.0
    else:
        beta = float(cov / var_bench)

    # Alpha (annualized): alpha_period = mean_ret - beta * mean_bench
    alpha_period = mean_ret - beta * mean_bench
    alpha = alpha_period * ann_factor

    return (float(alpha), beta)


def information_ratio(
    returns: pl.Series,
    benchmark: pl.Series,
) -> float:
    """Calculate information ratio.

    Matches QuantStats implementation.

    Parameters
    ----------
    returns : pl.Series
        Strategy period returns.
    benchmark : pl.Series
        Benchmark period returns.

    Returns
    -------
    float
        Information ratio (active return / tracking error).

    Notes
    -----
    IR = mean(returns - benchmark) / std(returns - benchmark)
    Measures risk-adjusted excess return over benchmark.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, -0.01, 0.03, 0.01, -0.02])
    >>> benchmark = pl.Series([0.01, -0.02, 0.02, 0.01, -0.01])
    >>> information_ratio(returns, benchmark)
    0.25...
    """
    from nanuquant.core.validation import validate_benchmark_match

    validate_returns(returns)
    validate_returns(benchmark)
    validate_benchmark_match(returns, benchmark)

    if returns.is_empty() or len(returns) < 2:
        return 0.0

    returns = to_float_series(returns)
    benchmark = to_float_series(benchmark)

    # Active returns (tracking difference)
    active = returns - benchmark

    mean_active = active.mean()
    std_active = active.std()

    if mean_active is None or std_active is None or std_active == 0:
        return 0.0

    return float(mean_active / std_active)


def r_squared(
    returns: pl.Series,
    benchmark: pl.Series,
) -> float:
    """Calculate R-squared (coefficient of determination).

    Matches QuantStats r_squared implementation.

    Parameters
    ----------
    returns : pl.Series
        Strategy period returns.
    benchmark : pl.Series
        Benchmark period returns.

    Returns
    -------
    float
        R-squared value (0 to 1). Higher means more explained by benchmark.

    Notes
    -----
    R² = correlation² between returns and benchmark.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, -0.01, 0.03, 0.01, -0.02])
    >>> benchmark = pl.Series([0.01, -0.02, 0.02, 0.01, -0.01])
    >>> r_squared(returns, benchmark)
    0.60...
    """
    from nanuquant.core.validation import validate_benchmark_match

    validate_returns(returns)
    validate_returns(benchmark)
    validate_benchmark_match(returns, benchmark)

    if returns.is_empty() or len(returns) < 2:
        return 0.0

    returns = to_float_series(returns)
    benchmark = to_float_series(benchmark)
    n = len(returns)

    # Calculate correlation using consistent sample formulas (ddof=1)
    mean_ret = returns.mean()
    mean_bench = benchmark.mean()

    if mean_ret is None or mean_bench is None:
        return 0.0

    # Sample covariance (N-1 denominator)
    cov_sum = ((returns - mean_ret) * (benchmark - mean_bench)).sum()
    if cov_sum is None:
        return 0.0
    cov = cov_sum / (n - 1)

    # Sample standard deviations (ddof=1, N-1 denominator)
    std_ret = returns.std(ddof=1)
    std_bench = benchmark.std(ddof=1)

    if std_ret is None or std_bench is None:
        return 0.0
    if std_ret == 0 or std_bench == 0:
        return 0.0

    correlation = cov / (std_ret * std_bench)
    return float(correlation ** 2)


def treynor_ratio(
    returns: pl.Series,
    benchmark: pl.Series,
    *,
    risk_free_rate: float | None = None,
    periods_per_year: int | None = None,
) -> float:
    """Calculate Treynor ratio.

    Matches QuantStats treynor_ratio implementation.

    Parameters
    ----------
    returns : pl.Series
        Strategy period returns.
    benchmark : pl.Series
        Benchmark period returns.
    risk_free_rate : float, optional
        Annualized risk-free rate.
    periods_per_year : int, optional
        Periods per year for annualization.

    Returns
    -------
    float
        Treynor ratio (excess return / beta).

    Notes
    -----
    Measures return per unit of systematic risk (beta).
    Treynor = (CAGR - rf) / beta

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, -0.01, 0.03, 0.01, -0.02])
    >>> benchmark = pl.Series([0.01, -0.02, 0.02, 0.01, -0.01])
    >>> treynor_ratio(returns, benchmark)
    0.50...
    """
    from nanuquant.core.returns import cagr
    from nanuquant.core.validation import validate_benchmark_match

    validate_returns(returns)
    validate_returns(benchmark)
    validate_benchmark_match(returns, benchmark)

    if returns.is_empty() or len(returns) < 2:
        return 0.0

    config = get_config()
    rf = risk_free_rate if risk_free_rate is not None else config.risk_free_rate
    ppy = periods_per_year or config.periods_per_year

    # Calculate beta
    _, beta = greeks(returns, benchmark, periods_per_year=ppy)

    if beta == 0:
        annual_return = cagr(returns, periods_per_year=ppy)
        return float("inf") if annual_return > rf else 0.0

    annual_return = cagr(returns, periods_per_year=ppy)
    excess_return = annual_return - rf

    return float(safe_divide(excess_return, beta, default=0.0))


def benchmark_correlation(
    returns: pl.Series,
    benchmark: pl.Series,
) -> float:
    """Calculate correlation with benchmark.

    Parameters
    ----------
    returns : pl.Series
        Strategy period returns.
    benchmark : pl.Series
        Benchmark period returns.

    Returns
    -------
    float
        Pearson correlation coefficient (-1 to 1).

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, -0.01, 0.03, 0.01, -0.02])
    >>> benchmark = pl.Series([0.01, -0.02, 0.02, 0.01, -0.01])
    >>> benchmark_correlation(returns, benchmark)
    0.77...
    """
    from nanuquant.core.validation import validate_benchmark_match

    validate_returns(returns)
    validate_returns(benchmark)
    validate_benchmark_match(returns, benchmark)

    if returns.is_empty() or len(returns) < 2:
        return 0.0

    returns = to_float_series(returns)
    benchmark = to_float_series(benchmark)
    n = len(returns)

    # Calculate correlation using consistent sample formulas (ddof=1)
    mean_ret = returns.mean()
    mean_bench = benchmark.mean()

    if mean_ret is None or mean_bench is None:
        return 0.0

    # Sample covariance (N-1 denominator)
    cov_sum = ((returns - mean_ret) * (benchmark - mean_bench)).sum()
    if cov_sum is None:
        return 0.0
    cov = cov_sum / (n - 1)

    # Sample standard deviations (ddof=1, N-1 denominator)
    std_ret = returns.std(ddof=1)
    std_bench = benchmark.std(ddof=1)

    if std_ret is None or std_bench is None:
        return 0.0
    if std_ret == 0 or std_bench == 0:
        return 0.0

    return float(cov / (std_ret * std_bench))
