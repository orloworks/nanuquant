"""Technical indicators on price series, as Polars expressions.

Every function accepts either a ``pl.Series`` of prices (returns a ``pl.Series``) or a
``pl.Expr`` (returns a ``pl.Expr``), so the same definition serves a single series and a
lazy, multi-column price frame:

>>> import polars as pl
>>> from nanuquant import indicators as ind
>>> prices = pl.Series("SPY", [100.0, 101.0, 102.5, 101.5, 103.0])
>>> ind.sma(prices, 3)  # doctest: +SKIP
>>> df = pl.DataFrame({"SPY": prices, "QQQ": prices * 2})
>>> df.select(ind.rsi(pl.col("SPY", "QQQ"), 2))  # doctest: +SKIP
>>> df.select(pl.col("SPY").ind.ema(3))  # doctest: +SKIP

Conventions
-----------
- Input is **prices**, not returns (unlike ``nanuquant.max_drawdown`` etc.).
- A missing price (null *or* NaN) is treated as missing. Output is null wherever the
  window is incomplete.
- Convention choices that differ between platforms are explicit keyword arguments,
  never silent defaults: ``percent=`` for units, ``annualize=`` for volatility, and
  ``period=None`` (all-time) vs an integer (rolling window) for drawdown peaks.
  Composer-compatible strategy engines pin these per indicator.
"""

from __future__ import annotations

from nanuquant.indicators.core import (
    cumulative_return,
    current_drawdown,
    current_price,
    ema,
    max_drawdown,
    moving_average_return,
    rsi,
    sma,
    std_dev_return,
)
from nanuquant.indicators.namespace import IndicatorsNamespace

__all__ = [
    "IndicatorsNamespace",
    "cumulative_return",
    "current_drawdown",
    "current_price",
    "ema",
    "max_drawdown",
    "moving_average_return",
    "rsi",
    "sma",
    "std_dev_return",
]
