"""Trade data processing and conversion module.

This module provides functions to convert trade data (buy/sell transactions)
into return series that can be consumed by nanuquant functions.

Examples
--------
>>> import polars as pl
>>> from nanuquant.trades import trades_to_returns, TradeResult
>>>
>>> trades = pl.DataFrame({
...     "entry_time": ["2024-01-01", "2024-01-02"],
...     "entry_price": [100.0, 105.0],
...     "exit_time": ["2024-01-02", "2024-01-03"],
...     "exit_price": [105.0, 103.0],
...     "direction": ["long", "long"],
... })
>>> result = trades_to_returns(trades)
>>>
>>> # Now use with any nanuquant function
>>> import nanuquant as pm
>>> sharpe = pm.sharpe(result.returns)
"""

from nanuquant.trades.config import (
    TradeConfig,
    get_trade_config,
    reset_trade_config,
    set_trade_config,
)
from nanuquant.trades.conversion import (
    calculate_single_trade_return,
    trades_to_returns,
)
from nanuquant.trades.equity import (
    build_equity_curve,
    build_equity_curve_no_mtm,
)
from nanuquant.trades.types import (
    AggregationMode,
    ReturnMethod,
    Trade,
    TradeDirection,
    TradeResult,
    TradeStatus,
)
from nanuquant.trades.validation import (
    InvalidDirectionError,
    InvalidPriceError,
    InvalidTradeDataError,
    InvalidTradeTimesError,
    MissingColumnError,
    validate_initial_capital,
    validate_min_trades,
    validate_prices_dataframe,
    validate_trade_dataframe,
    validate_trade_times,
)

__all__ = [
    # Main conversion function
    "trades_to_returns",
    "calculate_single_trade_return",
    # Equity curve
    "build_equity_curve",
    "build_equity_curve_no_mtm",
    # Types
    "Trade",
    "TradeResult",
    "TradeDirection",
    "TradeStatus",
    "ReturnMethod",
    "AggregationMode",
    # Configuration
    "TradeConfig",
    "get_trade_config",
    "set_trade_config",
    "reset_trade_config",
    # Validation
    "validate_trade_dataframe",
    "validate_trade_times",
    "validate_prices_dataframe",
    "validate_min_trades",
    "validate_initial_capital",
    # Exceptions
    "InvalidTradeDataError",
    "MissingColumnError",
    "InvalidPriceError",
    "InvalidDirectionError",
    "InvalidTradeTimesError",
]
