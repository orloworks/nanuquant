"""Frozen pandas reference for the TC engine's indicators (test-only).

Copied from orloworks/tc-trading-engine:
- ``tc-spec/engine/spec_evaluator.py::compute_indicator`` on ``feat/phase-0-prereqs``
  @ 74fb015 (2026-08-23): the Composer-converged semantics, e.g. ``std_dev_return``
  in daily (not annualized) percent and ``max_drawdown`` as worst-in-window.
- ``tc-spec/engine/indicators.py::_current_drawdown`` on ``main`` @ af1199e: the
  rolling-window ``current_drawdown`` (absent from the phase-0 dispatcher).

nanuquant.indicators must reproduce these values with the equivalent explicit
arguments (``percent=True`` etc.). The computations are copied unchanged; only the
engine's comments, docstrings and DataFrame plumbing were dropped. Keep them faithful
to the engine and do not "fix" them here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_indicator(series: pd.Series, ref: str, params: dict) -> pd.Series:
    period = int(params.get("period", 10))

    if ref == "current_price":
        return series

    elif ref == "sma":
        return series.rolling(window=period, min_periods=period).mean()

    elif ref == "rsi":
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    elif ref == "ema":
        return series.ewm(span=period, min_periods=period, adjust=False).mean()

    elif ref == "cumulative_return":
        return (series / series.shift(period) - 1.0) * 100.0

    elif ref == "max_drawdown":

        def _worst_dd_in_window(w: pd.Series) -> float:
            if w.isna().any() or len(w) == 0:
                return float("nan")
            running_peak = w.cummax()
            dd = (running_peak - w) / running_peak * 100.0
            return float(dd.max())

        return series.rolling(window=period, min_periods=period).apply(
            _worst_dd_in_window, raw=False
        )

    elif ref == "std_dev_return":
        daily_returns = series.pct_change()
        return daily_returns.rolling(window=period, min_periods=period).std() * 100.0

    elif ref == "moving_average_return":
        daily_returns = series.pct_change() * 100.0
        return daily_returns.rolling(window=period, min_periods=period).mean()

    else:
        raise ValueError(f"Unknown indicator ref: {ref}")


def current_drawdown_main(series: pd.Series, period: int) -> pd.Series:
    """``indicators.py::_current_drawdown`` on engine main (rolling-window peak, percent)."""
    rolling_max = series.rolling(window=period, min_periods=period).max()
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (rolling_max - series) / rolling_max * 100.0
    return result.where(rolling_max > 0, 0.0)
