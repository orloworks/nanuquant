"""Execution quality metrics.

This module provides metrics for measuring trading execution quality,
including implementation shortfall and market impact estimation.

References
----------
- Perold, A. F. (1988). "The Implementation Shortfall: Paper versus Reality"
- Almgren, R., et al. (2005). "Direct Estimation of Equity Market Impact"
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import polars as pl


class ImplementationShortfallResult(NamedTuple):
    """Result of implementation shortfall calculation.

    Attributes
    ----------
    total_shortfall : float
        Total implementation shortfall (in price units or basis points).
    delay_cost : float
        Cost due to delay between decision and execution start.
    trading_cost : float
        Cost due to price impact during execution.
    opportunity_cost : float
        Cost from unfilled orders (if any).
    shortfall_bps : float
        Total shortfall expressed in basis points.
    realized_pnl : float
        Realized P&L from the trade.
    paper_pnl : float
        Paper P&L (what would have been achieved at decision price).
    """

    total_shortfall: float
    delay_cost: float
    trading_cost: float
    opportunity_cost: float
    shortfall_bps: float
    realized_pnl: float
    paper_pnl: float


def implementation_shortfall(
    decision_price: float,
    execution_prices: pl.Series | list[float] | np.ndarray,
    quantities: pl.Series | list[float] | np.ndarray,
    side: int = 1,
    *,
    arrival_price: float | None = None,
    end_price: float | None = None,
    unfilled_quantity: float = 0.0,
) -> ImplementationShortfallResult:
    """Calculate implementation shortfall (IS) for a trade.

    Implementation shortfall measures the difference between the paper return
    (using decision price) and the actual return achieved after execution.

    Parameters
    ----------
    decision_price : float
        Price at the time of investment decision.
    execution_prices : pl.Series or array-like
        Prices at which each tranche was executed.
    quantities : pl.Series or array-like
        Quantity executed at each price. Must match length of execution_prices.
    side : int, default 1
        Trade direction: 1 for buy, -1 for sell.
    arrival_price : float, optional
        Price at start of execution. If None, uses first execution price.
    end_price : float, optional
        Price at end of trading. If None, uses last execution price.
    unfilled_quantity : float, default 0.0
        Quantity that was not filled.

    Returns
    -------
    ImplementationShortfallResult
        Named tuple containing:
        - total_shortfall: Total IS in price units
        - delay_cost: Cost from decision to arrival
        - trading_cost: Cost during execution
        - opportunity_cost: Cost from unfilled orders
        - shortfall_bps: IS in basis points
        - realized_pnl: Actual P&L achieved
        - paper_pnl: P&L at decision price

    Notes
    -----
    Implementation Shortfall = Paper Return - Actual Return

    For a buy order:
        IS = (Avg Execution Price - Decision Price) / Decision Price

    The shortfall is decomposed into:
    1. Delay Cost: Price move from decision to execution start
    2. Trading Cost: Price move during execution (market impact)
    3. Opportunity Cost: Missed gain from unfilled orders

    Examples
    --------
    >>> # Buy 1000 shares in two tranches
    >>> decision = 100.0
    >>> prices = [100.5, 101.0]
    >>> quantities = [500, 500]
    >>> result = implementation_shortfall(decision, prices, quantities, side=1)
    >>> result.shortfall_bps > 0  # Slippage occurred
    True

    References
    ----------
    Perold, A. F. (1988). "The Implementation Shortfall: Paper versus Reality"
    """
    # Convert to numpy arrays
    if isinstance(execution_prices, pl.Series):
        exec_prices = execution_prices.to_numpy()
    else:
        exec_prices = np.array(execution_prices, dtype=float)

    if isinstance(quantities, pl.Series):
        qtys = quantities.to_numpy()
    else:
        qtys = np.array(quantities, dtype=float)

    if len(exec_prices) != len(qtys):
        raise ValueError("execution_prices and quantities must have same length")

    if len(exec_prices) == 0:
        return ImplementationShortfallResult(
            total_shortfall=0.0,
            delay_cost=0.0,
            trading_cost=0.0,
            opportunity_cost=0.0,
            shortfall_bps=0.0,
            realized_pnl=0.0,
            paper_pnl=0.0,
        )

    if decision_price <= 0:
        raise ValueError("decision_price must be positive")

    # Arrival and end prices
    if arrival_price is None:
        arrival_price = exec_prices[0]
    if end_price is None:
        end_price = exec_prices[-1]

    # Calculate total executed quantity
    total_qty = np.sum(qtys)

    if total_qty == 0:
        return ImplementationShortfallResult(
            total_shortfall=0.0,
            delay_cost=0.0,
            trading_cost=0.0,
            opportunity_cost=0.0,
            shortfall_bps=0.0,
            realized_pnl=0.0,
            paper_pnl=0.0,
        )

    # Volume-weighted average execution price (VWAP)
    vwap = np.sum(exec_prices * qtys) / total_qty

    # Paper P&L (if executed at decision price)
    # For buy: Paper PnL = (Market Price - Decision Price) * Qty
    # For sell: Paper PnL = (Decision Price - Market Price) * Qty
    paper_pnl = side * (end_price - decision_price) * total_qty

    # Realized P&L
    realized_pnl = side * (end_price - vwap) * total_qty

    # Total implementation shortfall
    total_shortfall = paper_pnl - realized_pnl

    # In per-share terms: shortfall = side * (VWAP - decision_price)
    shortfall_per_share = side * (vwap - decision_price)

    # Shortfall in basis points
    shortfall_bps = (shortfall_per_share / decision_price) * 10000

    # Decomposition
    # Delay cost: price move from decision to arrival
    delay_cost = side * (arrival_price - decision_price) * total_qty

    # Trading cost: price move during execution (arrival to VWAP)
    trading_cost = side * (vwap - arrival_price) * total_qty

    # Opportunity cost: unfilled quantity
    if unfilled_quantity > 0:
        opportunity_cost = side * (end_price - decision_price) * unfilled_quantity
    else:
        opportunity_cost = 0.0

    # Total shortfall includes all components: executed + unfilled
    # Per Perold (1988): IS = delay_cost + trading_cost + opportunity_cost
    total_shortfall = float(total_shortfall) + opportunity_cost

    return ImplementationShortfallResult(
        total_shortfall=float(total_shortfall),
        delay_cost=float(delay_cost),
        trading_cost=float(trading_cost),
        opportunity_cost=float(opportunity_cost),
        shortfall_bps=float(shortfall_bps),
        realized_pnl=float(realized_pnl),
        paper_pnl=float(paper_pnl),
    )


def market_impact_estimate(
    trade_volume: float,
    avg_daily_volume: float,
    volatility: float,
    *,
    impact_coefficient: float = 0.1,
    exponent: float = 0.5,
) -> float:
    """Estimate market impact using the square-root law.

    The square-root law is a well-established empirical relationship
    between trade size and market impact.

    Parameters
    ----------
    trade_volume : float
        Volume of the trade in shares/contracts.
    avg_daily_volume : float
        Average daily trading volume.
    volatility : float
        Annualized volatility (as decimal, e.g., 0.20 for 20%).
    impact_coefficient : float, default 0.1
        Market impact coefficient (eta). Typical values: 0.05-0.15.
    exponent : float, default 0.5
        Power law exponent. Square-root law uses 0.5.

    Returns
    -------
    float
        Estimated price impact as a fraction (e.g., 0.005 = 0.5%).

    Notes
    -----
    The square-root market impact model:

        Impact = η * σ * (Q / V)^0.5

    Where:
    - η is the impact coefficient (typically 0.05-0.15)
    - σ is daily volatility
    - Q is trade volume
    - V is average daily volume

    The daily volatility is calculated as:
        σ_daily = σ_annual / sqrt(252)

    Examples
    --------
    >>> # Trade 100,000 shares when ADV is 1 million
    >>> impact = market_impact_estimate(
    ...     trade_volume=100_000,
    ...     avg_daily_volume=1_000_000,
    ...     volatility=0.20,
    ... )
    >>> impact > 0
    True
    >>> impact < 0.01  # Less than 1%
    True

    References
    ----------
    Almgren, R., et al. (2005). "Direct Estimation of Equity Market Impact"
    """
    if trade_volume < 0:
        raise ValueError("trade_volume must be non-negative")
    if avg_daily_volume <= 0:
        raise ValueError("avg_daily_volume must be positive")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")

    if trade_volume == 0:
        return 0.0

    # Convert annual volatility to daily
    daily_vol = volatility / math.sqrt(252)

    # Participation rate (fraction of ADV)
    participation = trade_volume / avg_daily_volume

    # Square-root market impact
    impact = impact_coefficient * daily_vol * (participation ** exponent)

    return float(impact)


def vwap_slippage(
    execution_prices: pl.Series | list[float] | np.ndarray,
    quantities: pl.Series | list[float] | np.ndarray,
    benchmark_vwap: float,
    side: int = 1,
) -> float:
    """Calculate VWAP slippage relative to a benchmark.

    Measures how much worse (or better) the execution was compared
    to a benchmark VWAP.

    Parameters
    ----------
    execution_prices : pl.Series or array-like
        Prices at which each tranche was executed.
    quantities : pl.Series or array-like
        Quantity executed at each price.
    benchmark_vwap : float
        Benchmark VWAP (e.g., market VWAP for the trading period).
    side : int, default 1
        Trade direction: 1 for buy, -1 for sell.

    Returns
    -------
    float
        Slippage in basis points. Positive = worse than benchmark.

    Examples
    --------
    >>> prices = [100.5, 101.0, 100.8]
    >>> quantities = [300, 400, 300]
    >>> benchmark = 100.6
    >>> slippage = vwap_slippage(prices, quantities, benchmark, side=1)
    """
    # Convert to numpy arrays
    if isinstance(execution_prices, pl.Series):
        exec_prices = execution_prices.to_numpy()
    else:
        exec_prices = np.array(execution_prices, dtype=float)

    if isinstance(quantities, pl.Series):
        qtys = quantities.to_numpy()
    else:
        qtys = np.array(quantities, dtype=float)

    if len(exec_prices) != len(qtys):
        raise ValueError("execution_prices and quantities must have same length")

    total_qty = np.sum(qtys)
    if total_qty == 0:
        return 0.0

    # Calculate execution VWAP
    exec_vwap = np.sum(exec_prices * qtys) / total_qty

    # Slippage in basis points
    # For buy: positive slippage if exec_vwap > benchmark (paid more)
    # For sell: positive slippage if exec_vwap < benchmark (received less)
    slippage = side * (exec_vwap - benchmark_vwap) / benchmark_vwap * 10000

    return float(slippage)


def spread_cost(
    bid: float | pl.Series,
    ask: float | pl.Series,
    mid: float | pl.Series | None = None,
) -> float:
    """Calculate spread cost as percentage of mid price.

    Parameters
    ----------
    bid : float or pl.Series
        Bid price(s).
    ask : float or pl.Series
        Ask price(s).
    mid : float or pl.Series, optional
        Mid price(s). If None, calculated as (bid + ask) / 2.

    Returns
    -------
    float
        Average spread cost in basis points.

    Examples
    --------
    >>> spread_cost(99.95, 100.05)  # 10 bps spread
    10.0
    """
    if isinstance(bid, pl.Series):
        bid_arr = bid.to_numpy()
    else:
        bid_arr = np.array([bid])

    if isinstance(ask, pl.Series):
        ask_arr = ask.to_numpy()
    else:
        ask_arr = np.array([ask])

    if mid is None:
        mid_arr = (bid_arr + ask_arr) / 2
    elif isinstance(mid, pl.Series):
        mid_arr = mid.to_numpy()
    else:
        mid_arr = np.array([mid])

    # Spread in bps
    spreads = (ask_arr - bid_arr) / mid_arr * 10000

    return float(np.mean(spreads))


def execution_vwap(
    execution_prices: pl.Series | list[float] | np.ndarray,
    quantities: pl.Series | list[float] | np.ndarray,
) -> float:
    """Calculate volume-weighted average price of execution.

    Parameters
    ----------
    execution_prices : pl.Series or array-like
        Prices at which each tranche was executed.
    quantities : pl.Series or array-like
        Quantity executed at each price.

    Returns
    -------
    float
        VWAP of the execution.

    Examples
    --------
    >>> prices = [100, 101, 102]
    >>> quantities = [100, 200, 100]
    >>> execution_vwap(prices, quantities)
    100.75
    """
    if isinstance(execution_prices, pl.Series):
        prices = execution_prices.to_numpy()
    else:
        prices = np.array(execution_prices, dtype=float)

    if isinstance(quantities, pl.Series):
        qtys = quantities.to_numpy()
    else:
        qtys = np.array(quantities, dtype=float)

    total_qty = np.sum(qtys)
    if total_qty == 0:
        return 0.0

    return float(np.sum(prices * qtys) / total_qty)
