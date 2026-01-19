"""Global configuration for nanuquant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from nanuquant.types import FrequencyType


@dataclass
class MetricsConfig:
    """Configuration for metrics calculations.

    Attributes
    ----------
    risk_free_rate : float
        Annualized risk-free rate for excess return calculations.
        Default is 0.0 (no risk-free adjustment).
    periods_per_year : int
        Number of trading periods per year for annualization.
        Default is 252 (daily trading days).
    frequency : FrequencyType
        Expected frequency of return data.
        Default is "D" (daily).
    var_confidence : float
        Confidence level for Value at Risk calculations.
        Default is 0.95 (95% confidence).
    rolling_window : int
        Default window size for rolling calculations.
        Default is 252 (1 year of daily data).
    mar : float
        Minimum Acceptable Return for Sortino ratio and related metrics.
        Default is 0.0.
    ddof : int
        Delta degrees of freedom for standard deviation calculations.
        Default is 1 (sample std dev).

    Examples
    --------
    >>> config = MetricsConfig(risk_free_rate=0.04, periods_per_year=252)
    >>> config.risk_free_rate
    0.04
    """

    risk_free_rate: float = 0.0
    periods_per_year: int = 252
    frequency: FrequencyType = "D"
    var_confidence: float = 0.95
    rolling_window: int = 252
    mar: float = 0.0
    ddof: int = 1


# Global default configuration - users can modify this
DEFAULT_CONFIG = MetricsConfig()


def get_config() -> MetricsConfig:
    """Get the current global configuration."""
    return DEFAULT_CONFIG


def set_config(config: MetricsConfig) -> None:
    """Set the global configuration.

    Parameters
    ----------
    config : MetricsConfig
        New configuration to use globally.
    """
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config


def reset_config() -> None:
    """Reset configuration to defaults."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = MetricsConfig()
