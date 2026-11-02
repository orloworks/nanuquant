"""Polars ``.ind`` expression namespace for price indicators.

>>> import polars as pl
>>> import nanuquant  # registers the namespace
>>> df = pl.DataFrame({"SPY": [100.0, 101.0, 102.5, 101.5, 103.0]})
>>> df.select(pl.col("SPY").ind.sma(3))  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from nanuquant.indicators import core


@pl.api.register_expr_namespace("ind")
class IndicatorsNamespace:
    """Price indicators on a Polars expression; see :mod:`nanuquant.indicators`."""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def current_price(self) -> pl.Expr:
        """See :func:`nanuquant.indicators.current_price`."""
        return core.current_price(self._expr)

    def sma(self, period: int) -> pl.Expr:
        """See :func:`nanuquant.indicators.sma`."""
        return core.sma(self._expr, period)

    def ema(self, period: int, *, adjust: bool = False) -> pl.Expr:
        """See :func:`nanuquant.indicators.ema`."""
        return core.ema(self._expr, period, adjust=adjust)

    def rsi(self, period: int, *, method: Literal["wilder"] = "wilder") -> pl.Expr:
        """See :func:`nanuquant.indicators.rsi`."""
        return core.rsi(self._expr, period, method=method)

    def cumulative_return(self, period: int, *, percent: bool = False) -> pl.Expr:
        """See :func:`nanuquant.indicators.cumulative_return`."""
        return core.cumulative_return(self._expr, period, percent=percent)

    def max_drawdown(self, period: int, *, percent: bool = False) -> pl.Expr:
        """See :func:`nanuquant.indicators.max_drawdown`."""
        return core.max_drawdown(self._expr, period, percent=percent)

    def current_drawdown(self, period: int | None = None, *, percent: bool = False) -> pl.Expr:
        """See :func:`nanuquant.indicators.current_drawdown`."""
        return core.current_drawdown(self._expr, period, percent=percent)

    def std_dev_return(
        self,
        period: int,
        *,
        percent: bool = False,
        annualize: bool = False,
        periods_per_year: int = 252,
    ) -> pl.Expr:
        """See :func:`nanuquant.indicators.std_dev_return`."""
        return core.std_dev_return(
            self._expr,
            period,
            percent=percent,
            annualize=annualize,
            periods_per_year=periods_per_year,
        )

    def moving_average_return(self, period: int, *, percent: bool = False) -> pl.Expr:
        """See :func:`nanuquant.indicators.moving_average_return`."""
        return core.moving_average_return(self._expr, period, percent=percent)
