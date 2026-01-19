"""Type definitions for nanuquant."""

from __future__ import annotations

from typing import Literal, Union

import polars as pl

# Input types that can be used for returns data
ReturnsInput = Union[pl.Series, pl.Expr]

# Supported frequency types for return data
FrequencyType = Literal["D", "W", "M", "H", "min", "s"]

# Annualization periods for each frequency
# D=daily (252 trading days), W=weekly, M=monthly, H=hourly (6.5h/day), min=minute
ANNUALIZATION_PERIODS: dict[FrequencyType, float] = {
    "D": 252.0,
    "W": 52.0,
    "M": 12.0,
    "H": 252.0 * 6.5,  # 1638 trading hours/year
    "min": 252.0 * 390.0,  # 98280 trading minutes/year
    "s": 252.0 * 390.0 * 60.0,  # trading seconds/year
}

# VaR confidence levels
VaRConfidence = Literal[0.90, 0.95, 0.99]

# Standard deviation multipliers for parametric VaR
VAR_SIGMA_MAP: dict[VaRConfidence, float] = {
    0.90: 1.282,
    0.95: 1.645,
    0.99: 2.326,
}

# Rolling window presets
RollingWindowType = Literal["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y"]

ROLLING_WINDOW_DAYS: dict[RollingWindowType, int] = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "2Y": 504,
    "3Y": 756,
    "5Y": 1260,
}
