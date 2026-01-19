"""Configuration for trade processing."""

from __future__ import annotations

from dataclasses import dataclass

from nanuquant.trades.types import AggregationMode, ReturnMethod


@dataclass
class TradeConfig:
    """Configuration for trade-to-returns conversion.

    Attributes
    ----------
    default_return_method : ReturnMethod
        Default method for calculating returns ("simple" or "log").
    default_aggregation : AggregationMode
        Default aggregation mode ("trade", "equity", "D", "W", "M").
    include_fees_default : bool
        Whether to include fees by default.
    fill_gaps_default : bool
        Whether to fill gaps in calendar aggregation by default.
    warn_on_missing_prices : bool
        Whether to warn when MTM is not possible due to missing prices.
    min_trades_for_metrics : int
        Minimum trades required for reliable metrics.
    """

    default_return_method: ReturnMethod = "simple"
    default_aggregation: AggregationMode = "trade"
    include_fees_default: bool = True
    fill_gaps_default: bool = True
    warn_on_missing_prices: bool = True
    min_trades_for_metrics: int = 30


# Module-level default config
_DEFAULT_TRADE_CONFIG = TradeConfig()


def get_trade_config() -> TradeConfig:
    """Get the current trade configuration."""
    return _DEFAULT_TRADE_CONFIG


def set_trade_config(config: TradeConfig) -> None:
    """Set the trade configuration.

    Parameters
    ----------
    config : TradeConfig
        New configuration to use.
    """
    global _DEFAULT_TRADE_CONFIG
    _DEFAULT_TRADE_CONFIG = config


def reset_trade_config() -> None:
    """Reset trade configuration to defaults."""
    global _DEFAULT_TRADE_CONFIG
    _DEFAULT_TRADE_CONFIG = TradeConfig()
