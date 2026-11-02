"""Missing-data policy: null and NaN are missing, never zero.

QuantStats fills missing returns with 0, turning a missing day into a flat day. nanuquant must
instead give the same answer as if the missing observations had never been there:

- single-series metrics equal the metric on the series with the missing rows removed;
- paired metrics equal the metric on the rows where both series are present;
- dated outputs drop a missing return together with its date;
- positional outputs keep their length and are null where the input is missing.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

import nanuquant as nq
from nanuquant.core import periods, timeseries
from nanuquant.institutional import (
    cornish_fisher_var,
    deflated_sharpe_ratio,
    downside_correlation,
    entropic_var,
    historical_var,
    lower_tail_dependence,
    parametric_var,
    probabilistic_sharpe_ratio,
    upside_correlation,
)
from nanuquant.institutional._helpers import newey_west_se
from nanuquant.institutional.robustness import minimum_track_record_length

N = 300
MISSING = [0, 7, 8, 9, 50, 151, 299]


def _clean(seed: int, mean: float = 0.0006) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(mean, 0.012, N)


def _with_missing(values: np.ndarray, missing: list[int], *, as_nan: bool) -> pl.Series:
    out = values.astype(float).copy().tolist()
    for i in missing:
        out[i] = float("nan") if as_nan else None
    return pl.Series("r", out, dtype=pl.Float64)


def _same(a: object, b: object) -> bool:
    if isinstance(a, tuple):
        return all(_same(x, y) for x, y in zip(a, b))  # type: ignore[arg-type]
    fa, fb = float(a), float(b)  # type: ignore[arg-type]
    if math.isnan(fa) and math.isnan(fb):
        return True
    return math.isclose(fa, fb, rel_tol=1e-12, abs_tol=1e-15)


SINGLE = {
    "comp": nq.comp,
    "cagr": nq.cagr,
    "avg_return": nq.avg_return,
    "avg_win": nq.avg_win,
    "avg_loss": nq.avg_loss,
    "best": nq.best,
    "worst": nq.worst,
    "win_rate": nq.win_rate,
    "payoff_ratio": nq.payoff_ratio,
    "profit_factor": nq.profit_factor,
    "consecutive_wins": nq.consecutive_wins,
    "consecutive_losses": nq.consecutive_losses,
    "volatility": nq.volatility,
    "var": nq.var,
    "cvar": nq.cvar,
    "max_drawdown": nq.max_drawdown,
    "ulcer_index": nq.ulcer_index,
    "downside_deviation": nq.downside_deviation,
    "sharpe": nq.sharpe,
    "sortino": nq.sortino,
    "calmar": nq.calmar,
    "omega": nq.omega,
    "gain_to_pain_ratio": nq.gain_to_pain_ratio,
    "ulcer_performance_index": nq.ulcer_performance_index,
    "kelly_criterion": nq.kelly_criterion,
    "tail_ratio": nq.tail_ratio,
    "common_sense_ratio": nq.common_sense_ratio,
    "risk_return_ratio": nq.risk_return_ratio,
    "recovery_factor": nq.recovery_factor,
    "skewness": nq.skewness,
    "kurtosis": nq.kurtosis,
    "jarque_bera": nq.jarque_bera,
    "shapiro_wilk": nq.shapiro_wilk,
    "outlier_win_ratio": nq.outlier_win_ratio,
    "outlier_loss_ratio": nq.outlier_loss_ratio,
    "expected_return": nq.expected_return,
    "geometric_mean": nq.geometric_mean,
    "exposure": nq.exposure,
    "ghpr": nq.ghpr,
    "rar": nq.rar,
    "cpc_index": nq.cpc_index,
    "serenity_index": nq.serenity_index,
    "risk_of_ruin": nq.risk_of_ruin,
    "adjusted_sortino": nq.adjusted_sortino,
    "smart_sharpe": nq.smart_sharpe,
    "smart_sortino": nq.smart_sortino,
    "sqn": nq.sqn,
    "expectancy": nq.expectancy,
    "k_ratio": nq.k_ratio,
    "psr": lambda r: probabilistic_sharpe_ratio(r).psr,
    "dsr": lambda r: deflated_sharpe_ratio(r, n_trials=10),
    "min_trl": minimum_track_record_length,
    "cornish_fisher_var": cornish_fisher_var,
    "entropic_var": entropic_var,
    "historical_var": historical_var,
    "parametric_var": parametric_var,
    "newey_west_se": newey_west_se,
}


@pytest.mark.parametrize("as_nan", [False, True], ids=["null", "nan"])
@pytest.mark.parametrize("name", list(SINGLE))
def test_single_series_metric_ignores_missing(name, as_nan):
    values = _clean(1)
    fn = SINGLE[name]
    with_missing = _with_missing(values, MISSING, as_nan=as_nan)
    clean = pl.Series("r", np.delete(values, MISSING))
    assert _same(fn(with_missing), fn(clean)), name


def test_missing_observations_are_not_zero_returns():
    # The QuantStats behaviour (fill with 0) would count the gaps as flat days.
    r = pl.Series([0.01, -0.02, None, 0.03, None, 0.01])
    assert nq.win_rate(r) == pytest.approx(0.75)
    assert nq.exposure(r) == pytest.approx(1.0)
    assert nq.cagr(r, periods_per_year=4) == pytest.approx(
        nq.cagr(pl.Series([0.01, -0.02, 0.03, 0.01]), periods_per_year=4)
    )


def test_all_missing_raises_empty():
    with pytest.raises(nq.EmptySeriesError):
        nq.sharpe(pl.Series([None, float("nan")], dtype=pl.Float64))


PAIRED = {
    "greeks": lambda r, b: nq.greeks(r, b),
    "information_ratio": nq.information_ratio,
    "r_squared": nq.r_squared,
    "treynor_ratio": nq.treynor_ratio,
    "benchmark_correlation": nq.benchmark_correlation,
    "lower_tail_dependence": lower_tail_dependence,
    "downside_correlation": downside_correlation,
    "upside_correlation": upside_correlation,
}


@pytest.mark.parametrize("name", list(PAIRED))
def test_paired_metric_uses_rows_where_both_present(name):
    r, b = _clean(2), _clean(3, mean=0.0004)
    r_missing, b_missing = [3, 40, 41], [5, 40, 200]
    fn = PAIRED[name]
    actual = fn(_with_missing(r, r_missing, as_nan=False), _with_missing(b, b_missing, as_nan=True))
    keep = np.ones(N, dtype=bool)
    keep[r_missing + b_missing] = False
    expected = fn(pl.Series(r[keep]), pl.Series(b[keep]))
    assert _same(actual, expected), name


def test_paired_metric_rejects_unequal_lengths():
    with pytest.raises(nq.BenchmarkMismatchError):
        nq.information_ratio(pl.Series([0.01, 0.02, None]), pl.Series([0.01, 0.02]))


def test_drawdown_series_keeps_positions():
    values = _clean(4)
    with_missing = _with_missing(values, MISSING, as_nan=True)
    dd = nq.to_drawdown_series(with_missing)
    assert dd.len() == N
    assert dd.is_null().arg_true().to_list() == MISSING
    clean_dd = nq.to_drawdown_series(pl.Series(np.delete(values, MISSING)))
    np.testing.assert_allclose(dd.drop_nulls().to_numpy(), clean_dd.to_numpy(), rtol=1e-12)


@pytest.mark.parametrize(
    "fn",
    [
        lambda r: nq.rolling_volatility(r, rolling_period=20),
        lambda r: nq.rolling_sharpe(r, rolling_period=20),
        lambda r: nq.rolling_sortino(r, rolling_period=20),
    ],
    ids=["volatility", "sharpe", "sortino"],
)
def test_rolling_metric_keeps_positions_and_nulls_windows_with_gaps(fn):
    values = _clean(5)
    with_missing = _with_missing(values, [150], as_nan=True)
    out = fn(with_missing)
    assert out.len() == N
    # windows that include position 150 are null; windows that don't are unaffected
    assert out[150:170].null_count() == 20
    full = fn(pl.Series(values))
    np.testing.assert_allclose(out[200:].to_numpy(), full[200:].to_numpy(), rtol=1e-12)


def test_rolling_beta_masks_both_series():
    r, b = _clean(6), _clean(7)
    out = nq.rolling_beta(
        _with_missing(r, [100], as_nan=False),
        _with_missing(b, [180], as_nan=True),
        rolling_period=10,
    )
    assert out.len() == N
    assert out[100:110].null_count() == 10
    assert out[180:190].null_count() == 10
    assert out[150] is not None


def _dates() -> pl.Series:
    return pl.Series("date", [date(2021, 1, 1) + timedelta(days=i) for i in range(N)])


def test_dated_outputs_drop_missing_with_their_dates():
    values = _clean(8)
    dates = _dates()
    with_missing = _with_missing(values, MISSING, as_nan=True)
    keep = np.ones(N, dtype=bool)
    keep[MISSING] = False
    clean, clean_dates = pl.Series(values[keep]), dates.filter(pl.Series(keep))
    assert timeseries.yearly_returns(with_missing, dates=dates).equals(
        timeseries.yearly_returns(clean, dates=clean_dates)
    )
    assert periods.monthly_returns(with_missing, dates=dates).equals(
        periods.monthly_returns(clean, dates=clean_dates)
    )
    assert timeseries.drawdown_details(with_missing, dates=dates, top_n=50).equals(
        timeseries.drawdown_details(clean, dates=clean_dates, top_n=50)
    )
    got = periods.distribution(with_missing, dates=dates)
    want = periods.distribution(clean, dates=clean_dates)
    for period in want:
        assert got[period]["values"] == pytest.approx(want[period]["values"])


def test_compare_aligns_rows_and_dates():
    r, b, dates = _clean(9), _clean(10), _dates()
    out = periods.compare(
        _with_missing(r, [1, 2], as_nan=False), _with_missing(b, [2, 3], as_nan=True), dates=dates
    )
    assert out.height == N - 3
    assert out["date"][1] == dates[4]


def test_dated_outputs_reject_length_mismatch():
    with pytest.raises(ValueError, match="dates length"):
        timeseries.yearly_returns(pl.Series([0.01, 0.02]), dates=_dates()[:3])


def _matrix_with_gaps() -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(11)
    full = pl.DataFrame({f"a{i}": rng.normal(0.0005, 0.01 + 0.002 * i, N) for i in range(4)})
    gaps = full.with_columns(
        pl.when(pl.int_range(pl.len()) == 10).then(None).otherwise(pl.col("a0")).alias("a0"),
        pl.when(pl.int_range(pl.len()) == 20)
        .then(float("nan"))
        .otherwise(pl.col("a2"))
        .alias("a2"),
    )
    keep = [i for i in range(N) if i not in (10, 20)]
    return gaps, full[keep]


def test_portfolio_functions_drop_incomplete_rows_instead_of_zero_filling():
    from nanuquant.institutional import (
        absorption_ratio,
        ledoit_wolf_covariance,
        marginal_contribution_to_risk,
        portfolio_volatility,
    )

    gaps, clean = _matrix_with_gaps()
    w = [0.4, 0.3, 0.2, 0.1]
    assert marginal_contribution_to_risk(gaps, w).mcr == pytest.approx(
        marginal_contribution_to_risk(clean, w).mcr
    )
    np.testing.assert_allclose(
        ledoit_wolf_covariance(gaps).covariance, ledoit_wolf_covariance(clean).covariance
    )
    assert portfolio_volatility(gaps, w) == pytest.approx(portfolio_volatility(clean, w))
    assert absorption_ratio(gaps).absorption_ratio == pytest.approx(
        absorption_ratio(clean).absorption_ratio
    )


def test_portfolio_functions_reject_too_few_complete_rows():
    from nanuquant.institutional import portfolio_volatility

    sparse = pl.DataFrame({"a": [0.01, None, 0.02], "b": [None, 0.01, 0.03]})
    with pytest.raises(ValueError, match="every asset present"):
        portfolio_volatility(sparse, [0.5, 0.5])


def test_total_loss_in_first_period_is_full_drawdown():
    dd = nq.to_drawdown_series(pl.Series([-1.0, 0.1]))
    assert dd.to_list() == [-1.0, -1.0]
    assert nq.max_drawdown(pl.Series([-1.0, 0.1])) == -1.0


def test_absorption_ratio_needs_as_many_complete_rows_as_assets():
    from nanuquant.institutional import absorption_ratio

    sparse = pl.DataFrame(
        {
            "a": [0.01, None, 0.02, None, 0.01],
            "b": [0.0, 0.01, None, 0.02, 0.01],
            "c": [0.01, 0.02, 0.03, None, -0.01],
        }
    )
    with pytest.raises(ValueError, match="complete observations"):
        absorption_ratio(sparse)


def test_drawdown_details_without_dates_reports_input_positions():
    from nanuquant.core.timeseries import drawdown_details

    r = pl.Series([None, None, 0.01, -0.05, -0.02, 0.10, 0.01])
    row = drawdown_details(r).row(0, named=True)
    dd = nq.to_drawdown_series(r)
    assert (row["start"], row["valley"], row["end"]) == (2, 4, 5)
    assert row["depth"] == pytest.approx(dd[4])


def test_drawdown_details_reports_first_period_total_loss():
    from nanuquant.core.timeseries import drawdown_details

    details = drawdown_details(pl.Series([-1.0, 0.0, 0.0]))
    assert details.height == 1
    assert details["depth"][0] == -1.0


def test_synthetic_dates_keep_each_return_on_its_own_day():
    # Without dates, return i falls on day i. Dropping a missing return must not slide
    # the later ones into the previous month.
    r = pl.Series([None] + [0.0] * 30 + [0.05])  # index 31 is 2020-02-01
    monthly = nq.monthly_returns(r)
    by_month = dict(zip(monthly["month"].to_list(), monthly["2020"].to_list(), strict=True))
    assert by_month[1] == pytest.approx(0.0)
    assert by_month[2] == pytest.approx(0.05)


def test_min_trl_of_a_constant_series_follows_the_sign_of_its_return():
    assert minimum_track_record_length(pl.Series([0.0] * 10)) == float("inf")
    assert minimum_track_record_length(pl.Series([-0.01] * 10)) == float("inf")
    assert minimum_track_record_length(pl.Series([0.01] * 10)) == 3.0  # ceil(1 + 1.645^2/2)


def test_rolling_metric_on_a_short_series_keeps_its_length():
    # map_batches inside the Polars namespace needs an output as long as the frame.
    df = pl.DataFrame({"r": [0.01, -0.02, 0.03]})
    out = df.with_columns(pl.col("r").metrics.rolling_sharpe(rolling_period=126).alias("s"))
    assert out["s"].null_count() == 3
    assert nq.rolling_volatility(pl.Series([0.01, 0.02]), rolling_period=5).len() == 2


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_cumulative_returns_carry_on_past_a_missing_return(missing):
    curve = timeseries.cumulative_returns(pl.Series([0.1, missing, -0.2, 0.05]))

    assert curve[1] is None
    assert curve[3] == pytest.approx(1.1 * 0.8 * 1.05 - 1)


def test_all_missing_series_is_missing_not_a_type_error():
    rolled = nq.rolling_volatility(pl.Series([None] * 5), rolling_period=2)

    assert len(rolled) == 5
    assert rolled.null_count() == 5
    assert nq.to_drawdown_series(pl.Series([None, None])).null_count() == 2
    with pytest.raises(nq.EmptySeriesError):
        nq.sharpe(pl.Series([None, None]))


def test_period_tables_are_chronological_for_unsorted_dates():
    dates = pl.Series([date(2021, 3, 1), date(2020, 1, 1), date(2020, 2, 1)])
    returns = pl.Series([0.03, 0.01, 0.02])
    assert periods.monthly_returns(returns, dates=dates).columns == ["month", "2020", "2021"]
    dist = periods.distribution(returns, dates=dates)
    assert dist["Daily"]["values"] == [0.01, 0.02, 0.03]
    assert dist["Monthly"]["values"] == pytest.approx([0.01, 0.02, 0.03])


def test_dated_outputs_reject_a_non_numeric_series_even_when_all_missing():
    with pytest.raises(TypeError):
        nq.yearly_returns(pl.Series([None, None], dtype=pl.String))
