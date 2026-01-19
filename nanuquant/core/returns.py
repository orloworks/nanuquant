"""Return metrics for nanuquant.

This module provides return-based metrics that match QuantStats output.
"""

from __future__ import annotations

import polars as pl

from nanuquant.config import get_config
from nanuquant.core.utils import (
    compound_returns,
    get_annualization_factor,
    safe_divide,
    to_float_series,
)
from nanuquant.core.validation import validate_min_length, validate_returns


def comp(returns: pl.Series) -> float:
    """Calculate total compounded return.

    Parameters
    ----------
    returns : pl.Series
        Period returns (not prices).

    Returns
    -------
    float
        Total compounded return. For example, 0.50 means 50% total return.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.10, 0.05, -0.02])
    >>> comp(returns)
    0.1319
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    return float((1 + returns).product() - 1)


def cagr(
    returns: pl.Series,
    *,
    periods_per_year: int | None = None,
) -> float:
    """Calculate Compound Annual Growth Rate.

    Parameters
    ----------
    returns : pl.Series
        Period returns (not prices).
    periods_per_year : int, optional
        Number of periods per year for annualization. If None, uses config default.

    Returns
    -------
    float
        Compound Annual Growth Rate. For example, 0.15 means 15% annual return.

    Notes
    -----
    Formula: (1 + total_return)^(periods_per_year / n) - 1

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01] * 252)  # 1% daily for a year
    >>> cagr(returns, periods_per_year=252)
    11.346...  # ~1134.6% annualized
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    n_periods = len(returns)

    if n_periods == 0:
        return 0.0

    config = get_config()
    ann_factor = get_annualization_factor(
        periods_per_year=periods_per_year or config.periods_per_year
    )

    total_return = comp(returns)
    # Handle negative total returns that would cause issues with fractional exponents
    if total_return <= -1:
        return float("-inf")

    years = n_periods / ann_factor
    if years == 0:
        return 0.0

    return float((1 + total_return) ** (1 / years) - 1)


def avg_return(returns: pl.Series) -> float:
    """Calculate average (arithmetic mean) return.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Arithmetic mean of returns.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, -0.01, 0.03])
    >>> avg_return(returns)
    0.0125
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    result = returns.mean()
    return float(result) if result is not None else 0.0


def avg_win(returns: pl.Series) -> float:
    """Calculate average winning (positive) return.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Average of positive returns. Returns 0.0 if no positive returns.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.03, -0.01])
    >>> avg_win(returns)
    0.02
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    wins = returns.filter(returns > 0)

    if wins.is_empty():
        return 0.0

    result = wins.mean()
    return float(result) if result is not None else 0.0


def avg_loss(returns: pl.Series) -> float:
    """Calculate average losing (negative) return.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Average of negative returns. Returns 0.0 if no negative returns.
        Note: This returns a negative value.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.03, -0.01])
    >>> avg_loss(returns)
    -0.015
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    losses = returns.filter(returns < 0)

    if losses.is_empty():
        return 0.0

    result = losses.mean()
    return float(result) if result is not None else 0.0


def best(returns: pl.Series) -> float:
    """Get the best (highest) single-period return.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Maximum return in the series.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.05, -0.01])
    >>> best(returns)
    0.05
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    result = returns.max()
    return float(result) if result is not None else 0.0


def worst(returns: pl.Series) -> float:
    """Get the worst (lowest) single-period return.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Minimum return in the series.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.05, -0.01])
    >>> worst(returns)
    -0.02
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    result = returns.min()
    return float(result) if result is not None else 0.0


def win_rate(returns: pl.Series) -> float:
    """Calculate percentage of positive returns.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Fraction of returns that are positive (0 to 1).

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.03, -0.01])
    >>> win_rate(returns)
    0.5
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    n_wins = (returns > 0).sum()
    return float(safe_divide(n_wins, len(returns), default=0.0))


def payoff_ratio(returns: pl.Series) -> float:
    """Calculate payoff ratio (average win / average loss).

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Ratio of average win to absolute average loss.
        Returns 0.0 if no wins or losses.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, -0.01, 0.03, -0.01])
    >>> payoff_ratio(returns)
    2.5
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    avg_w = avg_win(returns)
    avg_l = avg_loss(returns)

    # avg_loss returns negative, take absolute value
    if avg_l == 0:
        return float("inf") if avg_w > 0 else 0.0

    return float(safe_divide(avg_w, abs(avg_l), default=0.0))


def profit_factor(returns: pl.Series) -> float:
    """Calculate profit factor (sum of wins / sum of losses).

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    float
        Ratio of gross profits to gross losses.
        Returns inf if no losses, 0.0 if no wins.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.02, -0.01, 0.03, -0.01])
    >>> profit_factor(returns)
    2.5
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0.0

    returns = to_float_series(returns)
    wins = returns.filter(returns > 0)
    losses = returns.filter(returns < 0)

    sum_wins = wins.sum() if not wins.is_empty() else 0.0
    sum_losses = abs(losses.sum()) if not losses.is_empty() else 0.0

    if sum_losses == 0:
        return float("inf") if sum_wins > 0 else 0.0

    return float(safe_divide(sum_wins, sum_losses, default=0.0))


def consecutive_wins(returns: pl.Series) -> int:
    """Calculate maximum consecutive winning periods.

    Matches QuantStats consecutive_wins implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    int
        Maximum number of consecutive positive returns.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, 0.02, 0.03, -0.01, 0.02, 0.01])
    >>> consecutive_wins(returns)
    3
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0

    returns = to_float_series(returns)

    # Identify wins (1 for positive, 0 otherwise)
    is_win = (returns > 0).cast(pl.Int32)

    if is_win.sum() == 0:
        return 0

    # Count consecutive wins using cumsum and reset on loss
    # When we hit a loss, we mark a new group
    group_id = (is_win == 0).cum_sum()

    # Count wins within each group
    df = pl.DataFrame({"is_win": is_win, "group": group_id})

    # Filter only win rows and count per group
    win_groups = df.filter(pl.col("is_win") == 1).group_by("group").len()

    if win_groups.is_empty():
        return 0

    return int(win_groups["len"].max())


def consecutive_losses(returns: pl.Series) -> int:
    """Calculate maximum consecutive losing periods.

    Matches QuantStats consecutive_losses implementation.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    int
        Maximum number of consecutive negative returns.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, -0.03, -0.01, 0.02, 0.01])
    >>> consecutive_losses(returns)
    3
    """
    validate_returns(returns)
    if returns.is_empty():
        return 0

    returns = to_float_series(returns)

    # Identify losses (1 for negative, 0 otherwise)
    is_loss = (returns < 0).cast(pl.Int32)

    if is_loss.sum() == 0:
        return 0

    # Count consecutive losses using cumsum and reset on win
    # When we hit a win (or zero), we mark a new group
    group_id = (is_loss == 0).cum_sum()

    # Count losses within each group
    df = pl.DataFrame({"is_loss": is_loss, "group": group_id})

    # Filter only loss rows and count per group
    loss_groups = df.filter(pl.col("is_loss") == 1).group_by("group").len()

    if loss_groups.is_empty():
        return 0

    return int(loss_groups["len"].max())
