"""Indicator implementations.

Each public function builds a Polars expression from a price expression. ``_apply``
lets the same builder run on a ``pl.Series`` (eager) or a ``pl.Expr`` (lazy,
any number of columns).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable
from typing import Literal, TypeVar

import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

PriceInput = TypeVar("PriceInput", pl.Series, pl.Expr)

# Rows of windows processed per block in max_drawdown, bounding memory at
# _MDD_BLOCK * period floats regardless of series length.
_MDD_BLOCK = 4096


def _apply(prices: PriceInput, build: Callable[[pl.Expr], pl.Expr]) -> PriceInput:
    """Run ``build`` on a cleaned price expression, eagerly for a Series.

    NaN in the output (e.g. 0/0 from a zero price) is normalised to null, so every
    indicator reports a missing value one way.
    """
    if isinstance(prices, pl.Series):
        _require_numeric(prices)
        frame = pl.DataFrame({"_x": prices})
        out = build(_clean(pl.col("_x"))).fill_nan(None)
        return frame.select(out).to_series().alias(prices.name)
    if isinstance(prices, pl.Expr):
        return build(_clean(prices)).fill_nan(None)
    raise TypeError(f"prices must be a polars Series or Expr, got {type(prices).__name__}")


def _require_numeric(s: pl.Series) -> pl.Series:
    """Prices as Float64; non-numeric input raises (a Date column would otherwise be
    averaged as days-since-epoch)."""
    if not (s.dtype.is_numeric() or s.dtype == pl.Null):
        raise TypeError(f"prices must be numeric, got {s.dtype} for {s.name!r}")
    return s.cast(pl.Float64)


def _clean(x: pl.Expr) -> pl.Expr:
    """Float prices with NaN normalised to null, so every op treats both as missing.

    The type is checked when the expression runs (an expression doesn't know its
    column's dtype until then).
    """
    checked = x.map_batches(_require_numeric, return_dtype=pl.Float64, is_elementwise=True)
    return checked.fill_nan(None)


def _check_period(period: int, name: str = "period") -> int:
    """Validate a positive integer (numpy integers included) and return it as ``int``."""
    if isinstance(period, bool) or not isinstance(period, numbers.Integral) or period < 1:
        raise ValueError(f"{name} must be a positive integer, got {period!r}")
    return int(period)


def _scale(percent: bool) -> float:
    return 100.0 if percent else 1.0


def _simple_returns(x: pl.Expr) -> pl.Expr:
    """One-period simple returns; null wherever either price is missing (no padding)."""
    return x / x.shift(1) - 1.0


def current_price(prices: PriceInput) -> PriceInput:
    """Price itself, with NaN normalised to null.

    Parameters
    ----------
    prices : pl.Series or pl.Expr
        Price series.

    Returns
    -------
    pl.Series or pl.Expr
        Same kind as the input.
    """
    return _apply(prices, lambda x: x)


def sma(prices: PriceInput, period: int) -> PriceInput:
    """Simple moving average of price over ``period`` observations.

    Parameters
    ----------
    prices : pl.Series or pl.Expr
        Price series.
    period : int
        Window length. The first ``period - 1`` values are null.

    Returns
    -------
    pl.Series or pl.Expr
        Same kind as the input.
    """
    period = _check_period(period)
    return _apply(prices, lambda x: x.rolling_mean(window_size=period, min_samples=period))


def ema(prices: PriceInput, period: int, *, adjust: bool = False) -> PriceInput:
    """Exponential moving average with ``alpha = 2 / (period + 1)``.

    Parameters
    ----------
    prices : pl.Series or pl.Expr
        Price series.
    period : int
        Span. Output is null until ``period`` prices have been seen.
    adjust : bool, default False
        False is the recursive form seeded from the first price,
        ``y_t = (1 - alpha) * y_{t-1} + alpha * x_t``, over the full history (the
        Composer convention). True uses the finite-history weighted form.

    Returns
    -------
    pl.Series or pl.Expr
        Same kind as the input.
    """
    period = _check_period(period)
    return _apply(
        prices,
        lambda x: x.ewm_mean(span=period, adjust=adjust, min_samples=period, ignore_nulls=False),
    )


def rsi(prices: PriceInput, period: int, *, method: Literal["wilder"] = "wilder") -> PriceInput:
    """Relative Strength Index, 0 to 100.

    Parameters
    ----------
    prices : pl.Series or pl.Expr
        Price series.
    period : int
        Smoothing length; Wilder smoothing uses ``alpha = 1 / period``.
    method : {"wilder"}, default "wilder"
        Gains and losses are smoothed recursively from the first observation, with the
        first (undefined) change counted as zero. After a missing price the change is
        taken from the last valid price and the missing row is skipped, so a gap has
        the same effect as removing that row. There is no warm-up cut, so early
        values are defined but unsettled; callers needing a settled value should
        discard roughly ``3 * period`` observations.

    Returns
    -------
    pl.Series or pl.Expr
        Same kind as the input. Null where the price is missing or where both average
        gain and loss are zero (e.g. before the first price); 100 where only the
        average loss is zero.
    """
    period = _check_period(period)
    if method != "wilder":
        raise ValueError(f"unsupported RSI method {method!r}; only 'wilder' is implemented")
    alpha = 1.0 / period

    def build(x: pl.Expr) -> pl.Expr:
        # The change from the last valid price, so the move across a gap is measured
        # rather than lost; the first observation has no change and counts as zero.
        delta = x - x.forward_fill().shift(1)
        observed = x.is_not_null()
        gain = pl.when(observed).then(pl.when(delta > 0).then(delta).otherwise(0.0))
        loss = pl.when(observed).then(pl.when(delta < 0).then(-delta).otherwise(0.0))
        # A missing price is skipped in the smoothing, not counted as a flat day, so
        # RSI on a series with gaps equals RSI on the series with the gaps removed.
        avg_gain = gain.ewm_mean(alpha=alpha, adjust=False, min_samples=1, ignore_nulls=True)
        avg_loss = loss.ewm_mean(alpha=alpha, adjust=False, min_samples=1, ignore_nulls=True)
        rs = avg_gain / avg_loss
        # 100 - 100 / (1 + rs), written expression-first so the output keeps the
        # input column's name (a literal on the left would rename it "literal").
        value = (rs + 1.0).pow(-1.0) * -100.0 + 100.0
        return pl.when(observed).then(value)

    return _apply(prices, build)


def cumulative_return(prices: PriceInput, period: int, *, percent: bool = False) -> PriceInput:
    """Return over the last ``period`` observations: ``x_t / x_{t-period} - 1``.

    Parameters
    ----------
    prices : pl.Series or pl.Expr
        Price series.
    period : int
        Lookback in observations.
    percent : bool, default False
        If True, express in percent (x100), the Composer unit.

    Returns
    -------
    pl.Series or pl.Expr
        Same kind as the input.
    """
    period = _check_period(period)
    scale = _scale(percent)
    return _apply(prices, lambda x: (x / x.shift(period) - 1.0) * scale)


def _worst_drawdowns(s: pl.Series, period: int) -> pl.Series:
    """Worst peak-to-trough decline inside each trailing window, as a positive fraction."""
    arr: npt.NDArray[np.float64] = s.to_numpy()  # Float64 from _clean; null -> NaN
    out = np.full(arr.shape[0], np.nan)
    if arr.shape[0] >= period:
        windows = sliding_window_view(arr, period)
        for start in range(0, windows.shape[0], _MDD_BLOCK):
            w = windows[start : start + _MDD_BLOCK]
            peaks = np.maximum.accumulate(w, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                dd = (peaks - w) / peaks
            # A zero peak gives 0/0; skip those points rather than poison the window.
            worst = np.where(np.isnan(dd), -np.inf, dd).max(axis=1)
            worst[np.isneginf(worst) | np.isnan(w).any(axis=1)] = np.nan
            first = start + period - 1
            out[first : first + w.shape[0]] = worst
    return pl.Series(s.name, out, dtype=pl.Float64)


def max_drawdown(prices: PriceInput, period: int, *, percent: bool = False) -> PriceInput:
    """Worst peak-to-trough decline anywhere inside the trailing ``period`` window.

    The peak is the running maximum *within the window*, so a spike and subsequent
    drop mid-window counts even if the price has since recovered (the Composer
    definition). For distance from the peak to *today*, use :func:`current_drawdown`.

    Parameters
    ----------
    prices : pl.Series or pl.Expr
        Price series.
    period : int
        Window length. Any missing price in the window makes the result null.
    percent : bool, default False
        If True, express in percent (x100), the Composer unit.

    Returns
    -------
    pl.Series or pl.Expr
        Same kind as the input. A positive magnitude: 0.2 means a 20% decline.
    """
    period = _check_period(period)
    scale = _scale(percent)
    return _apply(
        prices,
        lambda x: (
            x.map_batches(lambda s: _worst_drawdowns(s, period), return_dtype=pl.Float64) * scale
        ),
    )


def current_drawdown(
    prices: PriceInput, period: int | None = None, *, percent: bool = False
) -> PriceInput:
    """Decline from the peak to the current price.

    Parameters
    ----------
    prices : pl.Series or pl.Expr
        Price series.
    period : int or None, default None
        None measures from the all-time running peak. An integer measures from the
        maximum of the trailing ``period`` window (null until the window is full).
    percent : bool, default False
        If True, express in percent (x100).

    Returns
    -------
    pl.Series or pl.Expr
        Same kind as the input. A positive magnitude: 0.1 means 10% below the peak.
    """
    if period is not None:
        period = _check_period(period)
    scale = _scale(percent)

    def build(x: pl.Expr) -> pl.Expr:
        if period is None:
            peak = x.cum_max()
        else:
            peak = x.rolling_max(window_size=period, min_samples=period)
        return (peak - x) / peak * scale

    return _apply(prices, build)


def std_dev_return(
    prices: PriceInput,
    period: int,
    *,
    percent: bool = False,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> PriceInput:
    """Sample standard deviation (ddof=1) of one-period returns over ``period`` returns.

    Parameters
    ----------
    prices : pl.Series or pl.Expr
        Price series.
    period : int
        Number of returns in the window (``period + 1`` prices).
    percent : bool, default False
        If True, express in percent (x100).
    annualize : bool, default False
        If True, multiply by ``sqrt(periods_per_year)``. Composer's unit is daily
        (not annualized).
    periods_per_year : int, default 252
        Used only when ``annualize`` is True.

    Returns
    -------
    pl.Series or pl.Expr
        Same kind as the input.
    """
    period = _check_period(period)
    periods_per_year = _check_period(periods_per_year, "periods_per_year")
    factor = _scale(percent) * (math.sqrt(periods_per_year) if annualize else 1.0)

    def build(x: pl.Expr) -> pl.Expr:
        r = _simple_returns(x)
        std = r.rolling_std(window_size=period, min_samples=period)
        # Polars' streaming variance leaves ~1e-10 residue on a constant window; a flat
        # window's dispersion is exactly zero, and threshold rules can depend on that.
        flat = r.rolling_min(window_size=period) == r.rolling_max(window_size=period)
        return std * (~flat).cast(pl.Float64) * factor

    return _apply(prices, build)


def moving_average_return(prices: PriceInput, period: int, *, percent: bool = False) -> PriceInput:
    """Arithmetic mean of one-period returns over ``period`` returns.

    Parameters
    ----------
    prices : pl.Series or pl.Expr
        Price series.
    period : int
        Number of returns in the window (``period + 1`` prices).
    percent : bool, default False
        If True, express in percent (x100), the Composer unit.

    Returns
    -------
    pl.Series or pl.Expr
        Same kind as the input.
    """
    period = _check_period(period)
    scale = _scale(percent)
    return _apply(
        prices,
        lambda x: _simple_returns(x).rolling_mean(window_size=period, min_samples=period) * scale,
    )
