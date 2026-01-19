"""Input validation for trade data."""

from __future__ import annotations

import polars as pl

from nanuquant.exceptions import EmptySeriesError, InsufficientDataError
from nanuquant.trades.types import (
    REQUIRED_PRICE_COLUMNS,
    REQUIRED_TRADE_COLUMNS,
)


class InvalidTradeDataError(Exception):
    """Raised when trade data is invalid or malformed."""

    def __init__(self, message: str, column: str | None = None) -> None:
        super().__init__(message)
        self.column = column


class MissingColumnError(InvalidTradeDataError):
    """Raised when a required column is missing."""

    def __init__(self, column: str, dataframe: str = "trades") -> None:
        super().__init__(
            f"Required column '{column}' is missing from {dataframe} DataFrame",
            column=column,
        )
        self.dataframe = dataframe


class InvalidPriceError(InvalidTradeDataError):
    """Raised when prices are invalid (negative, zero, NaN)."""

    def __init__(self, message: str = "Invalid price data") -> None:
        super().__init__(message)


class InvalidDirectionError(InvalidTradeDataError):
    """Raised when trade direction is invalid."""

    def __init__(self, value: str) -> None:
        super().__init__(f"Invalid direction '{value}': must be 'long' or 'short'")
        self.value = value


class InvalidTradeTimesError(InvalidTradeDataError):
    """Raised when trade times are invalid (exit before entry)."""

    def __init__(self, count: int) -> None:
        super().__init__(f"{count} trade(s) have exit_time <= entry_time")
        self.count = count


def validate_trade_dataframe(
    df: pl.DataFrame,
    *,
    require_exit: bool = False,
) -> None:
    """Validate trade DataFrame has required columns and valid data.

    Parameters
    ----------
    df : pl.DataFrame
        Trade data to validate.
    require_exit : bool, default False
        If True, require exit_time and exit_price columns.

    Raises
    ------
    EmptySeriesError
        If DataFrame is empty.
    MissingColumnError
        If required columns are missing.
    InvalidPriceError
        If prices contain invalid values.
    InvalidDirectionError
        If direction values are invalid.
    """
    if df.is_empty():
        raise EmptySeriesError("Trade DataFrame is empty")

    # Check required columns
    for col in REQUIRED_TRADE_COLUMNS:
        if col not in df.columns:
            raise MissingColumnError(col, "trades")

    if require_exit:
        if "exit_time" not in df.columns:
            raise MissingColumnError("exit_time", "trades")
        if "exit_price" not in df.columns:
            raise MissingColumnError("exit_price", "trades")

    # Validate prices are positive
    _validate_prices(df)

    # Validate direction if present
    if "direction" in df.columns:
        _validate_direction(df)


def validate_prices_dataframe(
    df: pl.DataFrame,
    *,
    require_symbol: bool = False,
) -> None:
    """Validate prices DataFrame has required columns and valid data.

    Parameters
    ----------
    df : pl.DataFrame
        Price data to validate.
    require_symbol : bool, default False
        If True, require symbol column for multi-asset support.

    Raises
    ------
    EmptySeriesError
        If DataFrame is empty.
    MissingColumnError
        If required columns are missing.
    InvalidPriceError
        If prices contain invalid values.
    """
    if df.is_empty():
        raise EmptySeriesError("Prices DataFrame is empty")

    # Check required columns
    for col in REQUIRED_PRICE_COLUMNS:
        if col not in df.columns:
            raise MissingColumnError(col, "prices")

    if require_symbol and "symbol" not in df.columns:
        raise MissingColumnError("symbol", "prices")

    # Validate close prices are positive
    close_series = df["close"]
    non_null = close_series.drop_nulls()
    if len(non_null) > 0 and (non_null <= 0).any():
        raise InvalidPriceError("Prices 'close' column contains non-positive values")


def validate_trade_times(df: pl.DataFrame) -> None:
    """Validate that exit times are after entry times.

    Parameters
    ----------
    df : pl.DataFrame
        Trade data with entry_time and exit_time columns.

    Raises
    ------
    InvalidTradeTimesError
        If any exit_time is before or equal to entry_time.
    """
    if "exit_time" not in df.columns:
        return

    # Filter to closed trades only
    closed = df.filter(pl.col("exit_time").is_not_null())

    if closed.is_empty():
        return

    invalid_count = closed.filter(pl.col("exit_time") <= pl.col("entry_time")).height

    if invalid_count > 0:
        raise InvalidTradeTimesError(invalid_count)


def validate_min_trades(df: pl.DataFrame, min_trades: int, metric: str = "") -> None:
    """Validate minimum number of trades for analysis.

    Parameters
    ----------
    df : pl.DataFrame
        Trade data.
    min_trades : int
        Minimum required number of trades.
    metric : str, optional
        Name of metric for error message.

    Raises
    ------
    InsufficientDataError
        If trade count is less than min_trades.
    """
    n_trades = df.height
    if n_trades < min_trades:
        raise InsufficientDataError(
            required=min_trades,
            actual=n_trades,
            metric=metric,
        )


def validate_initial_capital(capital: float | None, aggregation: str) -> None:
    """Validate initial capital for equity mode.

    Parameters
    ----------
    capital : float | None
        Initial capital value.
    aggregation : str
        Aggregation mode.

    Raises
    ------
    ValueError
        If capital is required but not provided or invalid.
    """
    if aggregation == "equity":
        if capital is None:
            raise ValueError("initial_capital is required for aggregation='equity'")
        if capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {capital}")


def _validate_prices(df: pl.DataFrame) -> None:
    """Validate price columns contain valid positive values."""
    price_cols = ["entry_price"]
    if "exit_price" in df.columns:
        price_cols.append("exit_price")

    for col in price_cols:
        if col not in df.columns:
            continue

        series = df[col]

        # Check for nulls in required fields
        if col == "entry_price" and series.null_count() > 0:
            raise InvalidPriceError(f"'{col}' contains null values")

        # Check for non-positive values (excluding nulls)
        non_null = series.drop_nulls()
        if len(non_null) > 0 and (non_null <= 0).any():
            raise InvalidPriceError(f"'{col}' contains non-positive values")


def _validate_direction(df: pl.DataFrame) -> None:
    """Validate direction column contains only valid values."""
    if "direction" not in df.columns:
        return

    valid_directions = {"long", "short"}
    unique_dirs = set(df["direction"].drop_nulls().unique().to_list())

    invalid = unique_dirs - valid_directions
    if invalid:
        raise InvalidDirectionError(str(invalid.pop()))
