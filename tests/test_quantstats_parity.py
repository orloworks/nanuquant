"""Strict parity with QuantStats (quantstats_lumi) on real market data.

Every function that documents QuantStats-compatible behaviour is checked at ``rtol=1e-9``
on the full SPY, QQQ and BND histories and on SPY with flat (zero-return) periods. The
few deliberate deviations are pinned by their exact relationship to QuantStats instead.
This replaces loose "finite and in range" checks that let formula drift go unnoticed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

import nanuquant as nq
from tests.conftest import load_cached_returns

qs = pytest.importorskip("quantstats_lumi", reason="quantstats_lumi required for parity tests")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
]

P = 252
STRICT = {"rtol": 1e-9, "atol": 1e-12}


def _datasets() -> dict[str, pd.Series]:
    data = {t: load_cached_returns(t) for t in ("SPY", "QQQ", "BND")}
    flat = data["SPY"].copy()
    flat.iloc[::7] = 0.0
    data["SPY_flat_days"] = flat
    return data


DATA = _datasets()

SINGLE = {
    "comp": (nq.comp, qs.stats.comp),
    "avg_return": (nq.avg_return, qs.stats.avg_return),
    "avg_win": (nq.avg_win, qs.stats.avg_win),
    "avg_loss": (nq.avg_loss, qs.stats.avg_loss),
    "best": (nq.best, qs.stats.best),
    "worst": (nq.worst, qs.stats.worst),
    "win_rate": (nq.win_rate, qs.stats.win_rate),
    "payoff_ratio": (nq.payoff_ratio, qs.stats.payoff_ratio),
    "profit_factor": (nq.profit_factor, qs.stats.profit_factor),
    "consecutive_wins": (nq.consecutive_wins, qs.stats.consecutive_wins),
    "consecutive_losses": (nq.consecutive_losses, qs.stats.consecutive_losses),
    "volatility": (
        lambda p: nq.volatility(p, periods_per_year=P),
        lambda r: qs.stats.volatility(r, periods=P),
    ),
    "var": (nq.var, qs.stats.var),
    "cvar": (nq.cvar, qs.stats.cvar),
    "max_drawdown": (nq.max_drawdown, qs.stats.max_drawdown),
    "ulcer_index": (nq.ulcer_index, qs.stats.ulcer_index),
    "sharpe": (lambda p: nq.sharpe(p, periods_per_year=P), lambda r: qs.stats.sharpe(r, periods=P)),
    "sortino": (
        lambda p: nq.sortino(p, periods_per_year=P),
        lambda r: qs.stats.sortino(r, periods=P),
    ),
    "smart_sharpe": (
        lambda p: nq.smart_sharpe(p, periods_per_year=P),
        lambda r: qs.stats.smart_sharpe(r, periods=P),
    ),
    "smart_sortino": (
        lambda p: nq.smart_sortino(p, periods_per_year=P),
        lambda r: qs.stats.smart_sortino(r, periods=P),
    ),
    "adjusted_sortino": (
        lambda p: nq.adjusted_sortino(p, periods_per_year=P),
        lambda r: qs.stats.adjusted_sortino(r, periods=P),
    ),
    "gain_to_pain_ratio": (nq.gain_to_pain_ratio, qs.stats.gain_to_pain_ratio),
    "ulcer_performance_index": (nq.ulcer_performance_index, qs.stats.ulcer_performance_index),
    "kelly_criterion": (nq.kelly_criterion, qs.stats.kelly_criterion),
    "tail_ratio": (nq.tail_ratio, qs.stats.tail_ratio),
    "common_sense_ratio": (nq.common_sense_ratio, qs.stats.common_sense_ratio),
    "risk_return_ratio": (nq.risk_return_ratio, qs.stats.risk_return_ratio),
    "recovery_factor": (nq.recovery_factor, qs.stats.recovery_factor),
    "skewness": (nq.skewness, qs.stats.skew),
    "kurtosis": (nq.kurtosis, qs.stats.kurtosis),
    "outlier_win_ratio": (nq.outlier_win_ratio, qs.stats.outlier_win_ratio),
    "outlier_loss_ratio": (nq.outlier_loss_ratio, qs.stats.outlier_loss_ratio),
    "expected_return": (nq.expected_return, qs.stats.expected_return),
    "geometric_mean": (nq.geometric_mean, qs.stats.geometric_mean),
    "ghpr": (nq.ghpr, qs.stats.ghpr),
    "cpc_index": (nq.cpc_index, qs.stats.cpc_index),
    "serenity_index": (nq.serenity_index, qs.stats.serenity_index),
    "risk_of_ruin": (nq.risk_of_ruin, qs.stats.risk_of_ruin),
}


@pytest.mark.parametrize("dataset", list(DATA))
@pytest.mark.parametrize("name", list(SINGLE))
def test_single_series_parity(name, dataset):
    returns = DATA[dataset]
    ours, theirs = SINGLE[name]
    np.testing.assert_allclose(ours(pl.Series(returns.to_numpy())), theirs(returns), **STRICT)


PAIRED = {
    "alpha": (
        lambda r, b: nq.greeks(r, b, periods_per_year=P)[0],
        lambda r, b: qs.stats.greeks(r, b, periods=P)["alpha"],
    ),
    "beta": (
        lambda r, b: nq.greeks(r, b, periods_per_year=P)[1],
        lambda r, b: qs.stats.greeks(r, b, periods=P)["beta"],
    ),
    "information_ratio": (nq.information_ratio, qs.stats.information_ratio),
    "r_squared": (nq.r_squared, qs.stats.r_squared),
}


@pytest.mark.parametrize("dataset", ["SPY", "BND", "SPY_flat_days"])
@pytest.mark.parametrize("name", list(PAIRED))
def test_paired_parity_against_qqq(name, dataset):
    common = DATA[dataset].index.intersection(DATA["QQQ"].index)
    r, b = DATA[dataset].loc[common], DATA["QQQ"].loc[common]
    ours, theirs = PAIRED[name]
    actual = ours(pl.Series(r.to_numpy()), pl.Series(b.to_numpy()))
    np.testing.assert_allclose(actual, theirs(r, b), **STRICT)


# ---------------------------------------------------------------------------
# Deliberate deviations, pinned by their exact relationship to QuantStats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", list(DATA))
def test_exposure_is_quantstats_without_round_up(dataset):
    returns = DATA[dataset]
    ours = nq.exposure(pl.Series(returns.to_numpy()))
    assert math.ceil(ours * 100) / 100 == pytest.approx(qs.stats.exposure(returns))
    assert ours == pytest.approx(float((returns != 0).mean()))


@pytest.mark.parametrize("dataset", list(DATA))
def test_cagr_uses_period_count_not_calendar_dates(dataset):
    returns = DATA[dataset]
    p = pl.Series(returns.to_numpy())
    expected = (1 + nq.comp(p)) ** (P / len(p)) - 1
    np.testing.assert_allclose(nq.cagr(p, periods_per_year=P), expected, **STRICT)
    # QuantStats measures years from the date index; the two agree closely on daily data
    np.testing.assert_allclose(nq.cagr(p, periods_per_year=P), qs.stats.cagr(returns), rtol=3e-3)


@pytest.mark.parametrize("dataset", list(DATA))
def test_rar_is_cagr_over_exact_exposure(dataset):
    p = pl.Series(DATA[dataset].to_numpy())
    expected = nq.cagr(p, periods_per_year=P) / nq.exposure(p)
    np.testing.assert_allclose(nq.rar(p, periods_per_year=P), expected, **STRICT)


@pytest.mark.parametrize("dataset", ["SPY", "BND"])
def test_treynor_annualizes_unlike_quantstats(dataset):
    common = DATA[dataset].index.intersection(DATA["QQQ"].index)
    r, b = (
        pl.Series(DATA[dataset].loc[common].to_numpy()),
        pl.Series(DATA["QQQ"].loc[common].to_numpy()),
    )
    beta = nq.greeks(r, b, periods_per_year=P)[1]
    np.testing.assert_allclose(
        nq.treynor_ratio(r, b, periods_per_year=P), nq.cagr(r, periods_per_year=P) / beta, **STRICT
    )
    # QuantStats divides the total multi-year return by beta instead
    qs_value = qs.stats.treynor_ratio(DATA[dataset].loc[common], DATA["QQQ"].loc[common])
    np.testing.assert_allclose(qs_value, nq.comp(r) / beta, rtol=1e-9)


@pytest.mark.parametrize("dataset", list(DATA))
def test_omega_matches_quantstats_formula(dataset):
    # quantstats_lumi.omega crashes on a Series (``.values`` on a scalar); this is its formula.
    r = DATA[dataset]
    threshold = (1 + 0.0) ** (1.0 / P) - 1
    excess = r - threshold
    expected = excess[excess > 0].sum() / -excess[excess < 0].sum()
    np.testing.assert_allclose(
        nq.omega(pl.Series(r.to_numpy()), periods_per_year=P), expected, **STRICT
    )


# ---------------------------------------------------------------------------
# Edge cases where nanuquant reports the mathematical limit
# ---------------------------------------------------------------------------


def test_kelly_without_wins_is_minus_infinity():
    assert nq.kelly_criterion(pl.Series([-0.01, -0.02, 0.0])) == float("-inf")


def test_kelly_without_losses_is_win_rate():
    assert nq.kelly_criterion(pl.Series([0.01, 0.02, 0.0])) == pytest.approx(1.0)


def test_kelly_without_any_nonzero_period_is_zero():
    # Fully in cash: no edge either way, so bet nothing.
    assert nq.kelly_criterion(pl.Series([0.0, 0.0, 0.0])) == 0.0


def test_omega_without_losses_is_infinite():
    assert nq.omega(pl.Series([0.01, 0.02, 0.03])) == float("inf")


def test_risk_of_ruin_cannot_overflow():
    losing = pl.Series([-0.01] * 9000 + [0.01] * 100)
    assert 0.0 <= nq.risk_of_ruin(losing) <= 1.0


def test_geometric_mean_of_total_loss_is_minus_one():
    assert nq.geometric_mean(pl.Series([0.1, -1.0, 0.2])) == -1.0
    assert nq.ghpr(pl.Series([0.1, -1.0, 0.2])) == -1.0


def test_geometric_mean_of_a_long_series_does_not_underflow_or_overflow():
    # 0.5 ** 1200 and 2 ** 1100 are out of float range as a product
    assert nq.geometric_mean(pl.Series([-0.5] * 1200)) == pytest.approx(-0.5, rel=1e-12)
    assert nq.geometric_mean(pl.Series([1.0] * 1100)) == pytest.approx(1.0, rel=1e-12)


def test_zero_exclusion_is_optional():
    r = pl.Series([0.01, 0.0, -0.01, 0.0])
    assert nq.win_rate(r) == pytest.approx(0.5)
    assert nq.win_rate(r, exclude_zeros=False) == pytest.approx(0.25)
    assert nq.avg_return(r) == pytest.approx(0.0)
    assert nq.avg_return(pl.Series([0.02, 0.0]), exclude_zeros=False) == pytest.approx(0.01)


ROLLING = {
    "rolling_volatility": (
        lambda p: nq.rolling_volatility(p, rolling_period=126, periods_per_year=365),
        lambda r: qs.stats.rolling_volatility(r, rolling_period=126, periods_per_year=365),
    ),
    "rolling_sharpe": (
        lambda p: nq.rolling_sharpe(p, rolling_period=126, periods_per_year=365),
        lambda r: qs.stats.rolling_sharpe(r, rolling_period=126, periods_per_year=365),
    ),
    "rolling_sortino": (
        lambda p: nq.rolling_sortino(p, rolling_period=126, periods_per_year=365),
        lambda r: qs.stats.rolling_sortino(r, rolling_period=126, periods_per_year=365),
    ),
}


@pytest.mark.parametrize("dataset", ["SPY", "BND"])
@pytest.mark.parametrize("name", list(ROLLING))
def test_rolling_parity_with_quantstats_annualization(name, dataset):
    returns = DATA[dataset]
    ours, theirs = ROLLING[name]
    got = ours(pl.Series(returns.to_numpy())).to_numpy().astype(float)
    want = theirs(returns).to_numpy(dtype=float)
    mask = ~np.isnan(want)
    assert np.array_equal(np.isnan(got), ~mask)
    np.testing.assert_allclose(got[mask], want[mask], rtol=1e-8, atol=1e-12)


def test_rolling_default_annualization_matches_core_metrics():
    p = pl.Series(DATA["SPY"].to_numpy()[-126:])
    last = nq.rolling_sharpe(p, rolling_period=126)[-1]
    assert last == pytest.approx(nq.sharpe(p), rel=1e-9)
