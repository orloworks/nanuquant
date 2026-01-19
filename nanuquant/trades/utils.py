"""Utility functions for trade processing."""

from __future__ import annotations

import math
from datetime import date, datetime

import polars as pl

from nanuquant.trades.types import (
    OPTIONAL_TRADE_COLUMNS,
    ReturnMethod,
)


def calculate_trade_return(
    entry_price: float,
    exit_price: float,
    *,
    direction: str = "long",
    fees: float = 0.0,
    quantity: float = 1.0,
    method: ReturnMethod = "simple",
) -> float:
    """Calculate return for a single trade.

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
        Trade quantity (absolute value).
    method : {"simple", "log"}, default "simple"
        Return calculation method.

    Returns
    -------
    float
        Trade return as decimal (e.g., 0.05 for 5%).

    Examples
    --------
    >>> calculate_trade_return(100.0, 105.0)
    0.05
    >>> calculate_trade_return(100.0, 95.0, direction="short")
    0.05
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if exit_price <= 0:
        raise ValueError("exit_price must be positive")

    # Calculate base return based on direction
    if direction == "long":
        if method == "simple":
            ret = (exit_price - entry_price) / entry_price
        else:  # log
            ret = math.log(exit_price / entry_price)
    else:  # short
        if method == "simple":
            ret = (entry_price - exit_price) / entry_price
        else:  # log
            ret = math.log(entry_price / exit_price)

    # Deduct fees as percentage of capital deployed
    if fees > 0 and quantity != 0:
        capital_deployed = entry_price * abs(quantity)
        fee_impact = fees / capital_deployed
        ret -= fee_impact

    return ret


def apply_column_defaults(df: pl.DataFrame) -> pl.DataFrame:
    """Apply default values for missing optional columns.

    Parameters
    ----------
    df : pl.DataFrame
        Trade DataFrame that may be missing optional columns.

    Returns
    -------
    pl.DataFrame
        DataFrame with all optional columns present (using defaults where missing).
    """
    result = df.clone()

    for col, (dtype, default) in OPTIONAL_TRADE_COLUMNS.items():
        if col not in result.columns and default is not None:
            result = result.with_columns(pl.lit(default).cast(dtype).alias(col))

    return result


def get_date_range(
    start: datetime | date,
    end: datetime | date,
) -> pl.Series:
    """Generate a series of dates between start and end (inclusive).

    Parameters
    ----------
    start : datetime or date
        Start date.
    end : datetime or date
        End date.

    Returns
    -------
    pl.Series
        Series of dates from start to end.
    """
    # Convert to date if datetime
    start_date = start.date() if isinstance(start, datetime) else start
    end_date = end.date() if isinstance(end, datetime) else end

    return pl.date_range(start_date, end_date, eager=True).alias("date")


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    quantity: float,
    direction: str,
    fees: float = 0.0,
) -> float:
    """Calculate profit/loss for a trade in absolute terms.

    Parameters
    ----------
    entry_price : float
        Entry price per unit.
    exit_price : float
        Exit price per unit.
    quantity : float
        Trade quantity.
    direction : {"long", "short"}
        Trade direction.
    fees : float, default 0.0
        Total transaction fees.

    Returns
    -------
    float
        Profit/loss in currency units.

    Examples
    --------
    >>> calculate_pnl(100.0, 105.0, 10, "long")
    50.0
    >>> calculate_pnl(100.0, 95.0, 10, "short")
    50.0
    """
    if direction == "long":
        pnl = (exit_price - entry_price) * quantity
    else:  # short
        pnl = (entry_price - exit_price) * quantity

    return pnl - fees


def calculate_unrealized_pnl(
    entry_price: float,
    current_price: float,
    quantity: float,
    direction: str,
) -> float:
    """Calculate unrealized P&L for an open position.

    Parameters
    ----------
    entry_price : float
        Entry price per unit.
    current_price : float
        Current market price.
    quantity : float
        Position quantity.
    direction : {"long", "short"}
        Position direction.

    Returns
    -------
    float
        Unrealized profit/loss in currency units.
    """
    if direction == "long":
        return (current_price - entry_price) * quantity
    else:  # short
        return (entry_price - current_price) * quantity


def datetime_to_date(dt: datetime | date) -> date:
    """Convert datetime to date, handling both types.

    Parameters
    ----------
    dt : datetime or date
        Input datetime or date.

    Returns
    -------
    date
        Date portion.
    """
    if isinstance(dt, datetime):
        return dt.date()
    return dt


def filter_closed_trades(df: pl.DataFrame) -> pl.DataFrame:
    """Filter DataFrame to only closed trades.

    Parameters
    ----------
    df : pl.DataFrame
        Trade DataFrame.

    Returns
    -------
    pl.DataFrame
        Only trades with exit_time and exit_price.
    """
    return df.filter(
        pl.col("exit_time").is_not_null() & pl.col("exit_price").is_not_null()
    )


def calculate_trade_returns_series(
    trades: pl.DataFrame,
    *,
    method: ReturnMethod = "simple",
    include_fees: bool = True,
) -> pl.DataFrame:
    """Calculate return for each trade in a DataFrame.

    Parameters
    ----------
    trades : pl.DataFrame
        Trade data with entry_price, exit_price, direction columns.
    method : {"simple", "log"}, default "simple"
        Return calculation method.
    include_fees : bool, default True
        Whether to deduct fees from returns.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with additional 'return' column.
    """
    # Ensure direction column exists with default
    if "direction" not in trades.columns:
        trades = trades.with_columns(pl.lit("long").alias("direction"))

    # Base return calculation depends on direction
    if method == "simple":
        return_expr = (
            pl.when(pl.col("direction") == "long")
            .then(
                (pl.col("exit_price") - pl.col("entry_price")) / pl.col("entry_price")
            )
            .otherwise(
                (pl.col("entry_price") - pl.col("exit_price")) / pl.col("entry_price")
            )
        )
    else:  # log returns
        return_expr = (
            pl.when(pl.col("direction") == "long")
            .then((pl.col("exit_price") / pl.col("entry_price")).log())
            .otherwise((pl.col("entry_price") / pl.col("exit_price")).log())
        )

    result = trades.with_columns(return_expr.alias("return"))

    # Deduct fees if requested
    if include_fees and "fees" in trades.columns and "quantity" in trades.columns:
        fee_impact = pl.col("fees") / (pl.col("entry_price") * pl.col("quantity").abs())
        result = result.with_columns((pl.col("return") - fee_impact).alias("return"))

    return result
