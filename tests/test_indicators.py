"""Tests for nanuquant.indicators.

The differential tests pin every indicator to the TC engine's frozen pandas reference
(tests/indicator_reference.py) on real SPY/QQQ/BND history and on synthetic series with
the edge cases strategy engines hit: a pre-inception gap and a flat run. Interior gaps are
pinned separately: there nanuquant deliberately departs from the reference, which pads
returns across the gap and keeps reporting EMA/RSI at the missing price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from nanuquant import indicators as ind
from tests.conftest import load_cached_returns
from tests.indicator_reference import compute_indicator, current_drawdown_main

TIGHT = 1e-8
PERIODS = [1, 2, 5, 10, 14, 20, 63, 200]


def _prices_from_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod() * 100.0


@pytest.fixture(scope="module")
def market_prices() -> dict[str, pd.Series]:
    return {t: _prices_from_returns(load_cached_returns(t)) for t in ("SPY", "QQQ", "BND")}


@pytest.fixture(scope="module")
def inception_prices() -> pd.Series:
    """Leading NaNs (pre-inception), then a random walk with a flat stretch."""
    rng = np.random.default_rng(42)
    walk = 50.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.03, 600))
    walk[300:320] = walk[299]
    return pd.Series(np.concatenate([np.full(40, np.nan), walk]))


def _assert_matches(actual: pl.Series, expected: pd.Series, *, label: str) -> None:
    """Null in ``actual`` must line up with NaN in ``expected``; values within TIGHT."""
    assert not actual.is_nan().any(), f"{label}: NaN leaked (missing must be null)"
    got = actual.to_numpy().astype(float)
    want = expected.to_numpy(dtype=float)
    assert got.shape == want.shape, label
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want), err_msg=f"{label}: missing mask")
    mask = ~np.isnan(want)
    np.testing.assert_allclose(got[mask], want[mask], rtol=TIGHT, atol=TIGHT, err_msg=label)


# (engine ref, nanuquant call with the engine's pinned conventions)
CASES = [
    ("current_price", lambda s, n: ind.current_price(s)),
    ("sma", lambda s, n: ind.sma(s, n)),
    ("ema", lambda s, n: ind.ema(s, n)),
    ("rsi", lambda s, n: ind.rsi(s, n)),
    ("cumulative_return", lambda s, n: ind.cumulative_return(s, n, percent=True)),
    ("max_drawdown", lambda s, n: ind.max_drawdown(s, n, percent=True)),
    ("std_dev_return", lambda s, n: ind.std_dev_return(s, n, percent=True)),
    ("moving_average_return", lambda s, n: ind.moving_average_return(s, n, percent=True)),
]


@pytest.mark.parametrize("ref,call", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("period", PERIODS)
def test_indicator_matches_engine_reference_on_market_data(market_prices, ref, call, period):
    if ref == "std_dev_return" and period == 1:
        pytest.skip("sample std of one return is undefined in both")
    for ticker, prices in market_prices.items():
        expected = compute_indicator(prices, ref, {"period": period})
        actual = call(pl.Series(ticker, prices.to_numpy()), period)
        _assert_matches(actual, expected, label=f"{ref}({period}) {ticker}")


@pytest.mark.parametrize("ref,call", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("period", [2, 10, 30])
def test_indicator_matches_engine_reference_across_inception_gap(
    inception_prices, ref, call, period
):
    expected = compute_indicator(inception_prices, ref, {"period": period})
    actual = call(pl.Series("x", inception_prices.to_numpy()), period)
    _assert_matches(actual, expected, label=f"{ref}({period}) inception")


@pytest.mark.parametrize("period", [5, 20, 63])
def test_rolling_current_drawdown_matches_engine_main_after_warmup(market_prices, period):
    for ticker, prices in market_prices.items():
        expected = current_drawdown_main(prices, period)
        actual = ind.current_drawdown(pl.Series(ticker, prices.to_numpy()), period, percent=True)
        # The engine returns 0.0 during warm-up (pandas .where on a NaN peak); nanuquant
        # returns null there, because "0% below the peak" is a false reading.
        _assert_matches(actual[period - 1 :], expected.iloc[period - 1 :], label=ticker)
        assert actual[: period - 1].null_count() == period - 1


def test_all_time_current_drawdown_matches_running_peak_definition():
    prices = pl.Series("x", [100.0, 120.0, 90.0, None, 130.0, 117.0])
    got = ind.current_drawdown(prices).to_list()
    assert got[:3] == pytest.approx([0.0, 0.0, 0.25])
    assert got[3] is None
    assert got[4:] == pytest.approx([0.0, 0.1])


def test_max_drawdown_counts_mid_window_spike_after_recovery():
    # Peak 130 mid-window, drop to 91 (30%), then recovery to 129: worst-in-window is
    # 30% even though the current gap from the peak is under 1%.
    prices = pl.Series("x", [100.0, 130.0, 91.0, 129.0])
    assert ind.max_drawdown(prices, 4)[-1] == pytest.approx(0.3)
    assert ind.current_drawdown(prices, 4)[-1] == pytest.approx(1 / 130)


def test_max_drawdown_spans_block_boundary():
    rng = np.random.default_rng(7)
    raw = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.02, 9000))
    expected = compute_indicator(pd.Series(raw), "max_drawdown", {"period": 50})
    _assert_matches(
        ind.max_drawdown(pl.Series("x", raw), 50, percent=True), expected, label="blocks"
    )


@pytest.mark.parametrize("ref,call", CASES, ids=[c[0] for c in CASES])
def test_expr_and_series_paths_agree_on_multiple_columns(market_prices, ref, call):
    spy = market_prices["SPY"].to_numpy()
    qqq = market_prices["QQQ"].to_numpy()
    n = min(len(spy), len(qqq))
    df = pl.DataFrame({"SPY": spy[:n], "QQQ": qqq[:n]})
    lazy = df.select(call(pl.col("SPY", "QQQ"), 10))
    assert lazy.columns == ["SPY", "QQQ"]
    for col in ("SPY", "QQQ"):
        eager = call(df[col], 10)
        assert eager.name == col
        assert lazy[col].equals(eager)


def test_namespace_matches_functions(market_prices):
    df = pl.DataFrame({"SPY": market_prices["SPY"].to_numpy()})
    via_namespace = df.select(
        pl.col("SPY").ind.rsi(10).alias("rsi"),
        pl.col("SPY").ind.max_drawdown(20, percent=True).alias("mdd"),
        pl.col("SPY").ind.std_dev_return(20, annualize=True).alias("vol"),
    )
    assert via_namespace["rsi"].equals(ind.rsi(df["SPY"], 10).alias("rsi"))
    assert via_namespace["mdd"].equals(ind.max_drawdown(df["SPY"], 20, percent=True).alias("mdd"))
    assert via_namespace["vol"].equals(
        ind.std_dev_return(df["SPY"], 20, annualize=True).alias("vol")
    )


def test_nan_and_null_are_both_missing():
    with_nan = pl.Series("x", [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0])
    with_null = pl.Series("x", [1.0, 2.0, None, 4.0, 5.0, 6.0])
    assert ind.sma(with_nan, 2).equals(ind.sma(with_null, 2))
    assert ind.sma(with_nan, 2).to_list() == [None, 1.5, None, None, 4.5, 5.5]


@pytest.mark.parametrize("ref,call", CASES, ids=[c[0] for c in CASES])
def test_interior_gap_is_missing_not_padded(ref, call):
    prices = pl.Series("x", [10.0, 11.0, 12.0, None, 13.0, 12.5, 14.0, 13.0, 15.0])
    out = call(prices, 2)
    assert out[3] is None, f"{ref}: a missing price must give a missing indicator"
    if ref in ("sma", "max_drawdown", "std_dev_return", "moving_average_return"):
        # Windows reaching across the gap stay missing (no padding over it).
        assert out[4] is None, ref
    assert out[-1] is not None, f"{ref}: must recover after the gap"


def test_zero_price_gives_null_not_nan():
    prices = pl.Series("x", [1.0, 0.0, 0.0, 2.0, 2.0])
    for _, call in CASES:
        assert not call(prices, 2).is_nan().any()


def test_numpy_integer_period_accepted():
    prices = pl.Series("x", [1.0, 2.0, 3.0, 4.0])
    assert ind.sma(prices, np.int64(2)).equals(ind.sma(prices, 2))


def test_rsi_extremes():
    rising = pl.Series("x", [1.0, 2.0, 3.0, 4.0])
    flat = pl.Series("x", [5.0, 5.0, 5.0])
    assert ind.rsi(rising, 2).to_list()[1:] == [100.0, 100.0, 100.0]
    assert ind.rsi(flat, 2).null_count() == 3


def test_std_dev_return_annualize_and_units():
    prices = pl.Series("x", [100.0, 101.0, 99.0, 102.0, 103.0, 101.0])
    daily = ind.std_dev_return(prices, 3)
    assert ind.std_dev_return(prices, 3, percent=True)[-1] == pytest.approx(daily[-1] * 100)
    assert ind.std_dev_return(prices, 3, annualize=True)[-1] == pytest.approx(
        daily[-1] * np.sqrt(252)
    )


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, None])
def test_invalid_period_raises(bad):
    with pytest.raises(ValueError, match="period"):
        ind.sma(pl.Series("x", [1.0, 2.0]), bad)


def test_unknown_rsi_method_raises():
    with pytest.raises(ValueError, match="wilder"):
        ind.rsi(pl.Series("x", [1.0, 2.0]), 2, method="cutler")  # type: ignore[arg-type]


def test_non_polars_input_raises():
    with pytest.raises(TypeError, match="Series or Expr"):
        ind.sma([1.0, 2.0, 3.0], 2)  # type: ignore[type-var]


def test_empty_series_returns_empty():
    empty = pl.Series("x", [], dtype=pl.Float64)
    for _, call in CASES:
        assert call(empty, 5).len() == 0


def test_rsi_with_an_interior_gap_equals_rsi_with_the_row_removed() -> None:
    prices = [100.0, 101.0, 100.5, 102.0, 101.0, 103.0, 104.0, 102.5, 103.5, 105.0]
    gappy = prices[:4] + [None] + prices[4:]
    with_gap = ind.rsi(pl.Series(gappy), 3)
    without = ind.rsi(pl.Series(prices), 3)
    assert with_gap[4] is None
    observed = with_gap.drop_nulls().to_list()
    assert observed == pytest.approx(without.drop_nulls().to_list(), rel=1e-12)


def test_non_numeric_prices_raise() -> None:
    from datetime import date

    df = pl.DataFrame({"d": [date(2024, 1, 1), date(2024, 1, 2)], "p": [1.0, 2.0]})
    with pytest.raises((TypeError, pl.exceptions.ComputeError)):
        df.select(ind.sma(pl.col("d"), 2))
    with pytest.raises(TypeError):
        ind.sma(df["d"], 2)
    assert df.select(ind.sma(pl.col("p"), 2))["p"].to_list() == [None, 1.5]
