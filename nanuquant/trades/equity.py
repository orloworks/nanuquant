"""Equity curve building with mark-to-market valuation."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl

from nanuquant.trades.utils import datetime_to_date, get_date_range


def build_equity_curve(
    trades: pl.DataFrame,
    initial_capital: float,
    prices: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build daily equity curve from trades with optional mark-to-market.

    Parameters
    ----------
    trades : pl.DataFrame
        Trade data with columns: entry_time, exit_time, entry_price,
        exit_price, quantity, direction, fees, symbol (optional).
    initial_capital : float
        Starting capital.
    prices : pl.DataFrame, optional
        Daily price data with columns: date, close, symbol (if multi-asset).
        Required for accurate mark-to-market valuation.

    Returns
    -------
    pl.DataFrame
        Daily equity curve with columns:
        - date: Calendar date
        - cash: Cash balance
        - position_value: Mark-to-market value of open positions
        - nav: Net Asset Value (cash + position_value)
        - daily_return: Day-over-day return

    Notes
    -----
    If prices is None, positions are valued at entry price until exit,
    and all P&L is recognized on the exit date.
    """
    # Determine date range
    start_date, end_date = _get_trade_date_range(trades)
    if start_date is None:
        # No trades, return empty curve
        return pl.DataFrame(
            {
                "date": [],
                "cash": [],
                "position_value": [],
                "nav": [],
                "daily_return": [],
            }
        ).cast(
            {
                "date": pl.Date,
                "cash": pl.Float64,
                "position_value": pl.Float64,
                "nav": pl.Float64,
                "daily_return": pl.Float64,
            }
        )

    # Generate all dates in range
    all_dates = get_date_range(start_date, end_date)

    # Check if we have multi-symbol trades
    has_symbol = "symbol" in trades.columns
    has_prices = prices is not None

    # Build price lookup if available
    price_lookup = None
    if has_prices:
        price_lookup = _build_price_lookup(prices, has_symbol)

    # Calculate daily NAV
    nav_data = []
    cash = initial_capital
    open_positions: list[dict] = []

    for current_date in all_dates:
        current_date_val = (
            current_date.date() if isinstance(current_date, datetime) else current_date
        )

        # Process entries for this date
        entries = _get_entries_on_date(trades, current_date_val)
        for entry in entries:
            cost = entry["entry_price"] * entry["quantity"]
            entry_fees = entry.get("fees", 0.0) / 2  # Split fees between entry/exit
            if entry.get("direction", "long") == "long":
                # Long: buy shares, cash decreases
                cash -= cost + entry_fees
            else:
                # Short: sell borrowed shares, cash increases (minus fees)
                cash += cost - entry_fees
            open_positions.append(entry)

        # Process exits for this date
        exits = _get_exits_on_date(trades, current_date_val)
        for exit_trade in exits:
            # Find matching open position
            pos_idx = _find_matching_position(open_positions, exit_trade)
            if pos_idx is not None:
                pos = open_positions.pop(pos_idx)
                exit_fees = pos.get("fees", 0.0) / 2
                if pos["direction"] == "long":
                    # Long: sell shares, cash increases
                    proceeds = exit_trade["exit_price"] * pos["quantity"]
                    cash += proceeds - exit_fees
                else:
                    # Short: buy back shares to cover, cash decreases
                    cover_cost = exit_trade["exit_price"] * pos["quantity"]
                    cash -= cover_cost + exit_fees

        # Mark-to-market open positions
        position_value = _calculate_position_value(
            open_positions, current_date_val, price_lookup, has_symbol
        )

        nav = cash + position_value
        nav_data.append(
            {
                "date": current_date_val,
                "cash": cash,
                "position_value": position_value,
                "nav": nav,
            }
        )

    # Create DataFrame and calculate returns
    result = pl.DataFrame(nav_data)

    # Calculate daily returns
    result = result.with_columns(
        (pl.col("nav") / pl.col("nav").shift(1) - 1).alias("daily_return")
    )

    # First day return is 0 (or based on initial capital)
    result = result.with_columns(pl.col("daily_return").fill_null(0.0))

    return result


def build_equity_curve_no_mtm(
    trades: pl.DataFrame,
    initial_capital: float,
) -> pl.DataFrame:
    """Build equity curve without mark-to-market (return on exit only).

    Parameters
    ----------
    trades : pl.DataFrame
        Trade data.
    initial_capital : float
        Starting capital.

    Returns
    -------
    pl.DataFrame
        Daily equity curve where returns are only recognized on exit dates.

    Notes
    -----
    This method is used when price data is not available. All trade P&L
    is recognized on the exit date, which understates intra-trade volatility.
    """
    # Get date range
    start_date, end_date = _get_trade_date_range(trades)
    if start_date is None:
        return pl.DataFrame(
            {
                "date": [],
                "cash": [],
                "position_value": [],
                "nav": [],
                "daily_return": [],
            }
        ).cast(
            {
                "date": pl.Date,
                "cash": pl.Float64,
                "position_value": pl.Float64,
                "nav": pl.Float64,
                "daily_return": pl.Float64,
            }
        )

    # Generate all dates
    all_dates = get_date_range(start_date, end_date)

    nav_data = []
    cash = initial_capital
    capital_deployed = 0.0

    for current_date in all_dates:
        current_date_val = (
            current_date.date() if isinstance(current_date, datetime) else current_date
        )

        # Process entries - capital is deployed but no P&L yet
        entries = _get_entries_on_date(trades, current_date_val)
        for entry in entries:
            cost = entry["entry_price"] * entry["quantity"]
            entry_fees = entry.get("fees", 0.0) / 2
            if entry.get("direction", "long") == "long":
                # Long: cash out, position value in
                cash -= cost + entry_fees
                capital_deployed += cost
            else:
                # Short: cash in (from sale), but we have a liability
                cash += cost - entry_fees
                capital_deployed -= cost  # Negative position value (liability)

        # Process exits - recognize full P&L
        exits = _get_exits_on_date(trades, current_date_val)
        for exit_trade in exits:
            entry_cost = exit_trade["entry_price"] * exit_trade["quantity"]
            exit_cost = exit_trade["exit_price"] * exit_trade["quantity"]
            exit_fees = exit_trade.get("fees", 0.0) / 2

            if exit_trade.get("direction", "long") == "long":
                # Long: sell shares, receive proceeds
                cash += exit_cost - exit_fees
                capital_deployed -= entry_cost
            else:
                # Short: buy back shares to cover
                cash -= exit_cost + exit_fees
                capital_deployed += entry_cost  # Remove the liability

        nav = cash + capital_deployed
        nav_data.append(
            {
                "date": current_date_val,
                "cash": cash,
                "position_value": capital_deployed,
                "nav": nav,
            }
        )

    result = pl.DataFrame(nav_data)
    result = result.with_columns(
        (pl.col("nav") / pl.col("nav").shift(1) - 1)
        .fill_null(0.0)
        .alias("daily_return")
    )

    return result


def _get_trade_date_range(
    trades: pl.DataFrame,
) -> tuple[date | None, date | None]:
    """Get the date range covered by trades."""
    if trades.is_empty():
        return None, None

    # Get min entry date and max exit date
    entry_dates = trades["entry_time"].cast(pl.Date)
    min_date = entry_dates.min()

    if "exit_time" in trades.columns:
        exit_dates = trades.filter(pl.col("exit_time").is_not_null())["exit_time"].cast(
            pl.Date
        )
        if not exit_dates.is_empty():
            max_date = max(entry_dates.max(), exit_dates.max())
        else:
            max_date = entry_dates.max()
    else:
        max_date = entry_dates.max()

    return min_date, max_date


def _build_price_lookup(
    prices: pl.DataFrame, has_symbol: bool
) -> dict[tuple[date, str | None], float]:
    """Build a price lookup dictionary for fast access."""
    lookup = {}

    for row in prices.iter_rows(named=True):
        price_date = row["date"]
        if isinstance(price_date, datetime):
            price_date = price_date.date()

        if has_symbol and "symbol" in prices.columns:
            key = (price_date, row.get("symbol"))
        else:
            key = (price_date, None)

        lookup[key] = row["close"]

    return lookup


def _get_entries_on_date(trades: pl.DataFrame, target_date: date) -> list[dict]:
    """Get all trade entries on a specific date."""
    entries = trades.filter(pl.col("entry_time").cast(pl.Date) == target_date)

    result = []
    for row in entries.iter_rows(named=True):
        result.append(
            {
                "entry_time": row["entry_time"],
                "entry_price": row["entry_price"],
                "quantity": row.get("quantity", 1.0),
                "direction": row.get("direction", "long"),
                "symbol": row.get("symbol"),
                "fees": row.get("fees", 0.0),
                "trade_id": row.get("trade_id"),
            }
        )

    return result


def _get_exits_on_date(trades: pl.DataFrame, target_date: date) -> list[dict]:
    """Get all trade exits on a specific date."""
    if "exit_time" not in trades.columns:
        return []

    exits = trades.filter(
        pl.col("exit_time").is_not_null()
        & (pl.col("exit_time").cast(pl.Date) == target_date)
    )

    result = []
    for row in exits.iter_rows(named=True):
        result.append(
            {
                "entry_time": row["entry_time"],
                "exit_time": row["exit_time"],
                "entry_price": row["entry_price"],
                "exit_price": row["exit_price"],
                "quantity": row.get("quantity", 1.0),
                "direction": row.get("direction", "long"),
                "symbol": row.get("symbol"),
                "fees": row.get("fees", 0.0),
                "trade_id": row.get("trade_id"),
            }
        )

    return result


def _find_matching_position(
    positions: list[dict], exit_trade: dict
) -> int | None:
    """Find the index of a matching open position for an exit.

    Matching priority:
    1. By trade_id if both positions have it
    2. By entry_time and symbol (FIFO - first matching position)

    Note: When multiple positions have the same entry_time and symbol,
    the first one (FIFO) will be matched. For unambiguous matching,
    provide unique trade_id values in your trade data.
    """
    for i, pos in enumerate(positions):
        # Match by trade_id if available (most reliable)
        if pos.get("trade_id") and exit_trade.get("trade_id"):
            if pos["trade_id"] == exit_trade["trade_id"]:
                return i
        # Otherwise match by entry_time and symbol (FIFO)
        elif (
            pos["entry_time"] == exit_trade["entry_time"]
            and pos.get("symbol") == exit_trade.get("symbol")
        ):
            return i

    return None


def _calculate_position_value(
    positions: list[dict],
    current_date: date,
    price_lookup: dict | None,
    has_symbol: bool,
) -> float:
    """Calculate total mark-to-market value of open positions."""
    total_value = 0.0

    for pos in positions:
        quantity = pos["quantity"]
        direction = pos.get("direction", "long")

        # Get current price
        if price_lookup is not None:
            symbol = pos.get("symbol") if has_symbol else None
            key = (current_date, symbol)
            current_price = price_lookup.get(key)

            if current_price is None:
                # Try to find most recent price
                current_price = _get_most_recent_price(
                    price_lookup, current_date, symbol
                )

            if current_price is None:
                # Fall back to entry price
                current_price = pos["entry_price"]
        else:
            # No prices available, use entry price
            current_price = pos["entry_price"]

        # Calculate position value
        if direction == "long":
            # Long: we own shares worth current_price * quantity
            total_value += current_price * quantity
        else:
            # Short: we owe shares, this is a liability (negative value)
            # The liability is what it would cost to cover: -current_price * quantity
            total_value -= current_price * quantity

    return total_value


def _get_most_recent_price(
    price_lookup: dict,
    target_date: date,
    symbol: str | None,
) -> float | None:
    """Get the most recent price on or before target date."""
    best_date = None
    best_price = None

    for (price_date, price_symbol), price in price_lookup.items():
        if price_symbol == symbol and price_date <= target_date:
            if best_date is None or price_date > best_date:
                best_date = price_date
                best_price = price

    return best_price
