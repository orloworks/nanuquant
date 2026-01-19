"""Type definitions for trade data processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import polars as pl

# Trade direction
TradeDirection = Literal["long", "short"]

# Trade status
TradeStatus = Literal["open", "closed"]

# Return calculation method
ReturnMethod = Literal["simple", "log"]

# Aggregation mode for converting trades to returns
AggregationMode = Literal["trade", "equity", "D", "W", "M"]

# Required columns for trade DataFrame input
REQUIRED_TRADE_COLUMNS: dict[str, pl.DataType] = {
    "entry_time": pl.Datetime,
    "entry_price": pl.Float64,
}

# Optional columns with their types and default values
OPTIONAL_TRADE_COLUMNS: dict[str, tuple[pl.DataType, object]] = {
    "exit_time": (pl.Datetime, None),
    "exit_price": (pl.Float64, None),
    "quantity": (pl.Float64, 1.0),
    "direction": (pl.Utf8, "long"),
    "symbol": (pl.Utf8, None),
    "fees": (pl.Float64, 0.0),
    "trade_id": (pl.Utf8, None),
}

# Required columns for prices DataFrame
REQUIRED_PRICE_COLUMNS: dict[str, pl.DataType] = {
    "date": pl.Date,
    "close": pl.Float64,
}

# Frequency mapping for aggregation
AGGREGATION_FREQUENCY_MAP: dict[str, str] = {
    "D": "1d",
    "W": "1w",
    "M": "1mo",
}


@dataclass
class Trade:
    """Single trade representation.

    Attributes
    ----------
    entry_time : datetime
        Trade entry timestamp.
    exit_time : datetime | None
        Trade exit timestamp. None for open trades.
    entry_price : float
        Entry price per unit.
    exit_price : float | None
        Exit price per unit. None for open trades.
    quantity : float
        Trade size (absolute value).
    direction : TradeDirection
        Trade direction: "long" or "short".
    symbol : str | None
        Optional ticker/symbol identifier.
    fees : float
        Total transaction costs (entry + exit).
    trade_id : str | None
        Optional unique trade identifier.
    """

    entry_time: datetime
    entry_price: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    quantity: float = 1.0
    direction: TradeDirection = "long"
    symbol: str | None = None
    fees: float = 0.0
    trade_id: str | None = None

    @property
    def is_closed(self) -> bool:
        """Return True if trade has been closed."""
        return self.exit_time is not None and self.exit_price is not None

    @property
    def status(self) -> TradeStatus:
        """Return trade status."""
        return "closed" if self.is_closed else "open"


@dataclass
class TradeResult:
    """Result of trade-to-returns conversion.

    Attributes
    ----------
    returns : pl.Series
        Period returns series.
    dates : pl.Series | None
        Corresponding dates (if available).
    n_trades : int
        Number of trades processed.
    n_winning : int
        Number of winning trades.
    n_losing : int
        Number of losing trades.
    total_fees : float
        Total fees across all trades.
    aggregation : str
        Aggregation mode used.
    periods_per_year : int
        Recommended annualization factor.
    has_mtm : bool
        Whether mark-to-market pricing was applied.
    warnings : list[str]
        Any warnings or caveats about the results.
    """

    returns: pl.Series
    dates: pl.Series | None = None
    n_trades: int = 0
    n_winning: int = 0
    n_losing: int = 0
    total_fees: float = 0.0
    aggregation: str = "trade"
    periods_per_year: int = 252
    has_mtm: bool = False
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure returns series has correct name."""
        if self.returns.name != "returns":
            self.returns = self.returns.alias("returns")
