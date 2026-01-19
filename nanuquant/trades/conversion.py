"""Core trades-to-returns conversion functions."""

from __future__ import annotations

import polars as pl

from nanuquant.trades.equity import (
    build_equity_curve,
    build_equity_curve_no_mtm,
)
from nanuquant.trades.types import (
    AGGREGATION_FREQUENCY_MAP,
    AggregationMode,
    ReturnMethod,
    TradeResult,
)
from nanuquant.trades.utils import (
    apply_column_defaults,
    calculate_trade_return,
    calculate_trade_returns_series,
    filter_closed_trades,
)
from nanuquant.trades.validation import (
    validate_initial_capital,
    validate_prices_dataframe,
    validate_trade_dataframe,
    validate_trade_times,
)


def trades_to_returns(
    trades: pl.DataFrame,
    *,
    prices: pl.DataFrame | None = None,
    aggregation: AggregationMode = "trade",
    initial_capital: float | None = None,
    method: ReturnMethod = "simple",
    include_fees: bool = True,
    fill_gaps: bool = True,
    trades_per_year: int | None = None,
) -> TradeResult:
    """Convert trade data to a returns series.

    This is the main entry point for converting trade data into the
    return series format expected by nanuquant functions.

    Parameters
    ----------
    trades : pl.DataFrame
        Trade data with columns:
        - entry_time (required): Trade entry timestamp
        - entry_price (required): Entry price per unit
        - exit_time: Exit timestamp (None for open trades)
        - exit_price: Exit price per unit
        - quantity: Trade size (default 1.0)
        - direction: "long" or "short" (default "long")
        - fees: Total transaction costs (default 0.0)
        - symbol: Ticker symbol (required for multi-asset with prices)
    prices : pl.DataFrame, optional
        Daily price data for mark-to-market valuation:
        - date (required): Calendar date
        - close (required): Closing price
        - symbol: Ticker symbol (if multi-asset)
        Required for accurate intra-trade daily returns.
    aggregation : {"trade", "equity", "D", "W", "M"}, default "trade"
        How to aggregate returns:
        - "trade": One return per closed trade
        - "equity": Daily returns from equity curve (requires initial_capital)
        - "D"/"W"/"M": Calendar-period returns
    initial_capital : float, optional
        Starting capital. Required for "equity" aggregation mode.
    method : {"simple", "log"}, default "simple"
        Return calculation method.
    include_fees : bool, default True
        Whether to deduct fees from returns.
    fill_gaps : bool, default True
        For calendar aggregation, fill periods without trades with 0% return.
    trades_per_year : int, optional
        Override periods_per_year for "trade" aggregation mode.
        Use this for proper annualization based on your trading frequency.

    Returns
    -------
    TradeResult
        Contains:
        - returns: The return series
        - dates: Corresponding dates (if available)
        - n_trades: Number of closed trades
        - n_winning/n_losing: Win/loss counts
        - total_fees: Total fees paid
        - aggregation: Mode used
        - periods_per_year: Annualization factor
        - has_mtm: Whether mark-to-market was applied
        - warnings: Any caveats about the results

    Examples
    --------
    >>> import polars as pl
    >>> trades = pl.DataFrame({
    ...     "entry_time": ["2024-01-01", "2024-01-02"],
    ...     "entry_price": [100.0, 105.0],
    ...     "exit_time": ["2024-01-02", "2024-01-03"],
    ...     "exit_price": [105.0, 103.0],
    ...     "direction": ["long", "long"],
    ... })

    >>> # Trade-level returns
    >>> result = trades_to_returns(trades)
    >>> result.returns
    shape: (2,)
    Series: 'returns' [f64]
    [
        0.05
        -0.019...
    ]

    >>> # Daily returns with mark-to-market
    >>> result = trades_to_returns(
    ...     trades,
    ...     prices=daily_prices,
    ...     aggregation="equity",
    ...     initial_capital=100000
    ... )

    Notes
    -----
    - For "equity" mode without prices, returns are recognized on exit
      date only. This understates intra-trade volatility.
    - Short trades: return = (entry - exit) / entry
    - Fees are deducted as: fee / (entry_price * quantity)
    """
    # Validate inputs
    validate_trade_dataframe(trades)
    validate_trade_times(trades)
    validate_initial_capital(initial_capital, aggregation)

    if prices is not None:
        has_symbol = "symbol" in trades.columns
        validate_prices_dataframe(prices, require_symbol=has_symbol)

    # Apply defaults for missing columns
    trades = apply_column_defaults(trades)

    # Handle fees
    if not include_fees and "fees" in trades.columns:
        trades = trades.with_columns(pl.lit(0.0).alias("fees"))

    # Route to appropriate aggregation handler
    if aggregation == "trade":
        return _aggregate_per_trade(trades, method, trades_per_year)
    elif aggregation == "equity":
        return _aggregate_equity(trades, prices, initial_capital, method)
    else:  # D, W, M
        return _aggregate_calendar(trades, prices, aggregation, initial_capital,
                                   method, fill_gaps)


def _aggregate_per_trade(
    trades: pl.DataFrame,
    method: ReturnMethod,
    trades_per_year: int | None,
) -> TradeResult:
    """Aggregate to one return per trade."""
    # Filter to closed trades
    closed = filter_closed_trades(trades)

    if closed.is_empty():
        return TradeResult(
            returns=pl.Series("returns", [], dtype=pl.Float64),
            n_trades=0,
            aggregation="trade",
            periods_per_year=trades_per_year or 252,
            warnings=["No closed trades found"],
        )

    # Calculate returns for each trade
    trade_returns = calculate_trade_returns_series(closed, method=method)

    returns = trade_returns["return"].alias("returns")
    dates = closed["exit_time"]

    # Calculate statistics
    n_trades = len(returns)
    n_winning = int((returns > 0).sum())
    n_losing = int((returns < 0).sum())
    total_fees = float(closed["fees"].sum()) if "fees" in closed.columns else 0.0

    # Determine periods_per_year
    warnings = []
    if trades_per_year is not None:
        periods = trades_per_year
    else:
        periods = 252  # Default to standard annualization
        warnings.append(
            "Using default periods_per_year=252 for trade-level returns. "
            "Set trades_per_year for accurate annualized metrics based on your trading frequency."
        )

    return TradeResult(
        returns=returns,
        dates=dates,
        n_trades=n_trades,
        n_winning=n_winning,
        n_losing=n_losing,
        total_fees=total_fees,
        aggregation="trade",
        periods_per_year=periods,
        has_mtm=False,
        warnings=warnings,
    )


def _aggregate_equity(
    trades: pl.DataFrame,
    prices: pl.DataFrame | None,
    initial_capital: float,
    method: ReturnMethod,
) -> TradeResult:
    """Aggregate to daily returns from equity curve."""
    warnings = []

    # Note: equity mode always returns simple returns from NAV changes
    if method == "log":
        warnings.append(
            "Equity mode returns simple returns (NAV-based), not log returns. "
            "The 'method' parameter only affects trade-level statistics."
        )

    # Build equity curve
    if prices is not None:
        equity = build_equity_curve(trades, initial_capital, prices)
        has_mtm = True
    else:
        equity = build_equity_curve_no_mtm(trades, initial_capital)
        has_mtm = False
        warnings.append(
            "Intra-trade volatility not captured. Returns recognized on exit only. "
            "Provide 'prices' DataFrame for accurate daily mark-to-market returns."
        )

    if equity.is_empty():
        return TradeResult(
            returns=pl.Series("returns", [], dtype=pl.Float64),
            n_trades=0,
            aggregation="equity",
            periods_per_year=252,
            has_mtm=has_mtm,
            warnings=warnings + ["No trades to process"],
        )

    # Get returns from equity curve
    returns = equity["daily_return"].alias("returns")
    dates = equity["date"]

    # Calculate trade statistics from original trades
    closed = filter_closed_trades(trades)
    n_trades = closed.height

    if n_trades > 0:
        trade_rets = calculate_trade_returns_series(closed, method=method)["return"]
        n_winning = int((trade_rets > 0).sum())
        n_losing = int((trade_rets < 0).sum())
        total_fees = float(closed["fees"].sum()) if "fees" in closed.columns else 0.0
    else:
        n_winning = 0
        n_losing = 0
        total_fees = 0.0

    return TradeResult(
        returns=returns,
        dates=dates,
        n_trades=n_trades,
        n_winning=n_winning,
        n_losing=n_losing,
        total_fees=total_fees,
        aggregation="equity",
        periods_per_year=252,
        has_mtm=has_mtm,
        warnings=warnings,
    )


def _aggregate_calendar(
    trades: pl.DataFrame,
    prices: pl.DataFrame | None,
    frequency: str,
    initial_capital: float | None,
    method: ReturnMethod,
    fill_gaps: bool,
) -> TradeResult:
    """Aggregate to calendar period returns (D/W/M)."""
    warnings = []

    # If we have prices and initial_capital, use equity curve approach
    if prices is not None and initial_capital is not None:
        equity = build_equity_curve(trades, initial_capital, prices)
        has_mtm = True
    elif initial_capital is not None:
        equity = build_equity_curve_no_mtm(trades, initial_capital)
        has_mtm = False
        warnings.append(
            "Intra-trade volatility not captured. Provide 'prices' for MTM valuation."
        )
    else:
        # Fall back to trade-level returns aggregated by period
        return _aggregate_trades_by_period(trades, frequency, method, fill_gaps)

    if equity.is_empty():
        return TradeResult(
            returns=pl.Series("returns", [], dtype=pl.Float64),
            n_trades=0,
            aggregation=frequency,
            periods_per_year=_get_periods_per_year(frequency),
            has_mtm=has_mtm,
            warnings=warnings + ["No trades to process"],
        )

    # Aggregate daily returns to desired frequency
    period_str = AGGREGATION_FREQUENCY_MAP[frequency]

    # Group by period and compound returns
    result = equity.group_by_dynamic("date", every=period_str).agg(
        # Compound returns: prod(1 + r) - 1
        ((1 + pl.col("daily_return")).product() - 1).alias("return")
    )

    returns = result["return"].alias("returns")
    dates = result["date"]

    # Calculate trade statistics
    closed = filter_closed_trades(trades)
    n_trades = closed.height

    if n_trades > 0:
        trade_rets = calculate_trade_returns_series(closed, method=method)["return"]
        n_winning = int((trade_rets > 0).sum())
        n_losing = int((trade_rets < 0).sum())
        total_fees = float(closed["fees"].sum()) if "fees" in closed.columns else 0.0
    else:
        n_winning = 0
        n_losing = 0
        total_fees = 0.0

    return TradeResult(
        returns=returns,
        dates=dates,
        n_trades=n_trades,
        n_winning=n_winning,
        n_losing=n_losing,
        total_fees=total_fees,
        aggregation=frequency,
        periods_per_year=_get_periods_per_year(frequency),
        has_mtm=has_mtm,
        warnings=warnings,
    )


def _aggregate_trades_by_period(
    trades: pl.DataFrame,
    frequency: str,
    method: ReturnMethod,
    fill_gaps: bool,
) -> TradeResult:
    """Aggregate trade returns by calendar period without equity curve."""
    closed = filter_closed_trades(trades)

    if closed.is_empty():
        return TradeResult(
            returns=pl.Series("returns", [], dtype=pl.Float64),
            n_trades=0,
            aggregation=frequency,
            periods_per_year=_get_periods_per_year(frequency),
            warnings=["No closed trades found"],
        )

    # Calculate per-trade returns
    trade_returns = calculate_trade_returns_series(closed, method=method)

    # Add exit date column
    trade_returns = trade_returns.with_columns(
        pl.col("exit_time").cast(pl.Date).alias("exit_date")
    )

    # Group by period
    period_str = AGGREGATION_FREQUENCY_MAP[frequency]

    result = trade_returns.group_by_dynamic("exit_date", every=period_str).agg(
        ((1 + pl.col("return")).product() - 1).alias("return")
    )

    returns = result["return"].alias("returns")
    dates = result["exit_date"]

    # Fill gaps if requested
    if fill_gaps and not result.is_empty():
        min_date = result["exit_date"].min()
        max_date = result["exit_date"].max()
        if min_date is not None and max_date is not None:
            full_date_range = pl.date_range(
                start=min_date,
                end=max_date,
                interval=period_str,
                eager=True,
            ).alias("exit_date").to_frame()
            result = full_date_range.join(
                result, on="exit_date", how="left"
            ).with_columns(pl.col("return").fill_null(0.0))
            returns = result["return"].alias("returns")
            dates = result["exit_date"]

    # Trade statistics
    n_trades = closed.height
    trade_rets = trade_returns["return"]
    n_winning = int((trade_rets > 0).sum())
    n_losing = int((trade_rets < 0).sum())
    total_fees = float(closed["fees"].sum()) if "fees" in closed.columns else 0.0

    return TradeResult(
        returns=returns,
        dates=dates,
        n_trades=n_trades,
        n_winning=n_winning,
        n_losing=n_losing,
        total_fees=total_fees,
        aggregation=frequency,
        periods_per_year=_get_periods_per_year(frequency),
        has_mtm=False,
        warnings=["Returns aggregated by exit date. Use equity mode for accurate period returns."],
    )


def _get_periods_per_year(frequency: str) -> int:
    """Get annualization factor for frequency."""
    return {"D": 252, "W": 52, "M": 12}.get(frequency, 252)


# Convenience function for single trade calculation
def calculate_single_trade_return(
    entry_price: float,
    exit_price: float,
    *,
    direction: str = "long",
    fees: float = 0.0,
    quantity: float = 1.0,
    method: ReturnMethod = "simple",
) -> float:
    """Calculate return for a single trade.

    Convenience function for calculating individual trade returns
    without creating a DataFrame.

    Parameters
    ----------
    entry_price : float
        Entry price per unit.
    exit_price : float
        Exit price per unit.
    direction : {"long", "short"}, default "long"
        Trade direction.
    fees : float, default 0.0
        Total transaction fees.
    quantity : float, default 1.0
        Trade quantity.
    method : {"simple", "log"}, default "simple"
        Return calculation method.

    Returns
    -------
    float
        Trade return as decimal.

    Examples
    --------
    >>> calculate_single_trade_return(100.0, 105.0)
    0.05
    >>> calculate_single_trade_return(100.0, 95.0, direction="short")
    0.05
    """
    return calculate_trade_return(
        entry_price,
        exit_price,
        direction=direction,
        fees=fees,
        quantity=quantity,
        method=method,
    )
