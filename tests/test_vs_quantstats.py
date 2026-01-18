"""Differential tests comparing polars_metrics against QuantStats.

These tests verify that our implementations match QuantStats output exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
import quantstats_lumi as qs

import polars_metrics as pm
from polars_metrics.exceptions import EmptySeriesError

# Tolerance levels from implementation plan
EXACT = {"rtol": 1e-10, "atol": 0}
TIGHT = {"rtol": 1e-8, "atol": 1e-12}
LOOSE = {"rtol": 1e-6, "atol": 1e-10}
STAT = {"rtol": 1e-4, "atol": 1e-8}

# For metrics that depend on calendar-based year calculations
# QuantStats uses 365.25 day years, so we use 365 periods for synthetic daily data
CALENDAR_PERIODS = 365


class TestReturnsVsQuantStats:
    """Test return metrics against QuantStats."""

    def test_comp_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test total compounded return on synthetic data."""
        expected = qs.stats.comp(sample_returns)
        actual = pm.comp(polars_returns)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_comp_spy(self, spy_returns: pd.Series, spy_polars: pl.Series) -> None:
        """Test total compounded return on SPY data."""
        expected = qs.stats.comp(spy_returns)
        actual = pm.comp(spy_polars)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_comp_qqq(self, qqq_returns: pd.Series, qqq_polars: pl.Series) -> None:
        """Test total compounded return on QQQ data."""
        expected = qs.stats.comp(qqq_returns)
        actual = pm.comp(qqq_polars)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_comp_bnd(self, bnd_returns: pd.Series, bnd_polars: pl.Series) -> None:
        """Test total compounded return on BND data."""
        expected = qs.stats.comp(bnd_returns)
        actual = pm.comp(bnd_polars)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    def test_cagr_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test CAGR on synthetic data.

        Note: QuantStats uses calendar-based years (index dates), so we use
        periods_per_year=365 to match calendar day interpretation.
        """
        expected = qs.stats.cagr(sample_returns, periods=CALENDAR_PERIODS)
        actual = pm.cagr(polars_returns, periods_per_year=CALENDAR_PERIODS)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_cagr_spy(self, spy_returns: pd.Series, spy_polars: pl.Series) -> None:
        """Test CAGR on SPY data."""
        expected = qs.stats.cagr(spy_returns, periods=CALENDAR_PERIODS)
        actual = pm.cagr(spy_polars, periods_per_year=CALENDAR_PERIODS)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_cagr_qqq(self, qqq_returns: pd.Series, qqq_polars: pl.Series) -> None:
        """Test CAGR on QQQ data."""
        expected = qs.stats.cagr(qqq_returns, periods=CALENDAR_PERIODS)
        actual = pm.cagr(qqq_polars, periods_per_year=CALENDAR_PERIODS)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    def test_avg_return_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test average return on synthetic data."""
        expected = qs.stats.avg_return(sample_returns)
        actual = pm.avg_return(polars_returns)
        np.testing.assert_allclose(actual, expected, **EXACT)

    @pytest.mark.integration
    def test_avg_return_spy(
        self, spy_returns: pd.Series, spy_polars: pl.Series
    ) -> None:
        """Test average return on SPY data."""
        expected = qs.stats.avg_return(spy_returns)
        actual = pm.avg_return(spy_polars)
        np.testing.assert_allclose(actual, expected, **EXACT)

    def test_avg_win_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test average win on synthetic data."""
        expected = qs.stats.avg_win(sample_returns)
        actual = pm.avg_win(polars_returns)
        np.testing.assert_allclose(actual, expected, **EXACT)

    @pytest.mark.integration
    def test_avg_win_spy(self, spy_returns: pd.Series, spy_polars: pl.Series) -> None:
        """Test average win on SPY data."""
        expected = qs.stats.avg_win(spy_returns)
        actual = pm.avg_win(spy_polars)
        np.testing.assert_allclose(actual, expected, **EXACT)

    def test_avg_loss_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test average loss on synthetic data."""
        expected = qs.stats.avg_loss(sample_returns)
        actual = pm.avg_loss(polars_returns)
        np.testing.assert_allclose(actual, expected, **EXACT)

    @pytest.mark.integration
    def test_avg_loss_spy(self, spy_returns: pd.Series, spy_polars: pl.Series) -> None:
        """Test average loss on SPY data."""
        expected = qs.stats.avg_loss(spy_returns)
        actual = pm.avg_loss(spy_polars)
        np.testing.assert_allclose(actual, expected, **EXACT)

    def test_best_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test best return on synthetic data."""
        expected = qs.stats.best(sample_returns)
        actual = pm.best(polars_returns)
        np.testing.assert_allclose(actual, expected, **EXACT)

    @pytest.mark.integration
    def test_best_spy(self, spy_returns: pd.Series, spy_polars: pl.Series) -> None:
        """Test best return on SPY data."""
        expected = qs.stats.best(spy_returns)
        actual = pm.best(spy_polars)
        np.testing.assert_allclose(actual, expected, **EXACT)

    def test_worst_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test worst return on synthetic data."""
        expected = qs.stats.worst(sample_returns)
        actual = pm.worst(polars_returns)
        np.testing.assert_allclose(actual, expected, **EXACT)

    @pytest.mark.integration
    def test_worst_spy(self, spy_returns: pd.Series, spy_polars: pl.Series) -> None:
        """Test worst return on SPY data."""
        expected = qs.stats.worst(spy_returns)
        actual = pm.worst(spy_polars)
        np.testing.assert_allclose(actual, expected, **EXACT)


class TestRiskVsQuantStats:
    """Test risk metrics against QuantStats."""

    def test_volatility_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test volatility on synthetic data."""
        expected = qs.stats.volatility(sample_returns, periods=252)
        actual = pm.volatility(polars_returns, periods_per_year=252)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_volatility_spy(
        self, spy_returns: pd.Series, spy_polars: pl.Series
    ) -> None:
        """Test volatility on SPY data."""
        expected = qs.stats.volatility(spy_returns, periods=252)
        actual = pm.volatility(spy_polars, periods_per_year=252)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_volatility_qqq(
        self, qqq_returns: pd.Series, qqq_polars: pl.Series
    ) -> None:
        """Test volatility on QQQ data."""
        expected = qs.stats.volatility(qqq_returns, periods=252)
        actual = pm.volatility(qqq_polars, periods_per_year=252)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_volatility_bnd(
        self, bnd_returns: pd.Series, bnd_polars: pl.Series
    ) -> None:
        """Test volatility on BND data."""
        expected = qs.stats.volatility(bnd_returns, periods=252)
        actual = pm.volatility(bnd_polars, periods_per_year=252)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    def test_var_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test VaR on synthetic data."""
        expected = qs.stats.var(sample_returns, confidence=0.95)
        actual = pm.var(polars_returns, confidence=0.95)
        np.testing.assert_allclose(actual, expected, **LOOSE)

    @pytest.mark.integration
    def test_var_spy(self, spy_returns: pd.Series, spy_polars: pl.Series) -> None:
        """Test VaR on SPY data."""
        expected = qs.stats.var(spy_returns, confidence=0.95)
        actual = pm.var(spy_polars, confidence=0.95)
        np.testing.assert_allclose(actual, expected, **LOOSE)

    def test_cvar_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test CVaR on synthetic data."""
        expected = qs.stats.cvar(sample_returns, confidence=0.95)
        actual = pm.cvar(polars_returns, confidence=0.95)
        np.testing.assert_allclose(actual, expected, **LOOSE)

    @pytest.mark.integration
    def test_cvar_spy(self, spy_returns: pd.Series, spy_polars: pl.Series) -> None:
        """Test CVaR on SPY data."""
        expected = qs.stats.cvar(spy_returns, confidence=0.95)
        actual = pm.cvar(spy_polars, confidence=0.95)
        np.testing.assert_allclose(actual, expected, **LOOSE)

    def test_max_drawdown_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test max drawdown on synthetic data."""
        expected = qs.stats.max_drawdown(sample_returns)
        actual = pm.max_drawdown(polars_returns)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_max_drawdown_spy(
        self, spy_returns: pd.Series, spy_polars: pl.Series
    ) -> None:
        """Test max drawdown on SPY data."""
        expected = qs.stats.max_drawdown(spy_returns)
        actual = pm.max_drawdown(spy_polars)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_max_drawdown_qqq(
        self, qqq_returns: pd.Series, qqq_polars: pl.Series
    ) -> None:
        """Test max drawdown on QQQ data."""
        expected = qs.stats.max_drawdown(qqq_returns)
        actual = pm.max_drawdown(qqq_polars)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_max_drawdown_bnd(
        self, bnd_returns: pd.Series, bnd_polars: pl.Series
    ) -> None:
        """Test max drawdown on BND data."""
        expected = qs.stats.max_drawdown(bnd_returns)
        actual = pm.max_drawdown(bnd_polars)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    def test_to_drawdown_series_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test drawdown series on synthetic data."""
        expected = qs.stats.to_drawdown_series(sample_returns).values
        actual = pm.to_drawdown_series(polars_returns).to_numpy()
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_to_drawdown_series_spy(
        self, spy_returns: pd.Series, spy_polars: pl.Series
    ) -> None:
        """Test drawdown series on SPY data."""
        expected = qs.stats.to_drawdown_series(spy_returns).values
        actual = pm.to_drawdown_series(spy_polars).to_numpy()
        np.testing.assert_allclose(actual, expected, **TIGHT)


class TestPerformanceVsQuantStats:
    """Test performance metrics against QuantStats."""

    def test_sharpe_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test Sharpe ratio on synthetic data."""
        expected = qs.stats.sharpe(sample_returns, periods=252, rf=0.0)
        actual = pm.sharpe(polars_returns, periods_per_year=252, risk_free_rate=0.0)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_sharpe_spy(self, spy_returns: pd.Series, spy_polars: pl.Series) -> None:
        """Test Sharpe ratio on SPY data."""
        expected = qs.stats.sharpe(spy_returns, periods=252, rf=0.0)
        actual = pm.sharpe(spy_polars, periods_per_year=252, risk_free_rate=0.0)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_sharpe_qqq(self, qqq_returns: pd.Series, qqq_polars: pl.Series) -> None:
        """Test Sharpe ratio on QQQ data."""
        expected = qs.stats.sharpe(qqq_returns, periods=252, rf=0.0)
        actual = pm.sharpe(qqq_polars, periods_per_year=252, risk_free_rate=0.0)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_sharpe_bnd(self, bnd_returns: pd.Series, bnd_polars: pl.Series) -> None:
        """Test Sharpe ratio on BND data."""
        expected = qs.stats.sharpe(bnd_returns, periods=252, rf=0.0)
        actual = pm.sharpe(bnd_polars, periods_per_year=252, risk_free_rate=0.0)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_sharpe_with_rf(
        self, spy_returns: pd.Series, spy_polars: pl.Series
    ) -> None:
        """Test Sharpe ratio with non-zero risk-free rate."""
        expected = qs.stats.sharpe(spy_returns, periods=252, rf=0.04)
        actual = pm.sharpe(spy_polars, periods_per_year=252, risk_free_rate=0.04)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    def test_sortino_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test Sortino ratio on synthetic data."""
        expected = qs.stats.sortino(sample_returns, periods=252, rf=0.0)
        actual = pm.sortino(polars_returns, periods_per_year=252, risk_free_rate=0.0)
        np.testing.assert_allclose(actual, expected, **LOOSE)

    @pytest.mark.integration
    def test_sortino_spy(self, spy_returns: pd.Series, spy_polars: pl.Series) -> None:
        """Test Sortino ratio on SPY data."""
        expected = qs.stats.sortino(spy_returns, periods=252, rf=0.0)
        actual = pm.sortino(spy_polars, periods_per_year=252, risk_free_rate=0.0)
        np.testing.assert_allclose(actual, expected, **LOOSE)

    @pytest.mark.integration
    def test_sortino_qqq(self, qqq_returns: pd.Series, qqq_polars: pl.Series) -> None:
        """Test Sortino ratio on QQQ data."""
        expected = qs.stats.sortino(qqq_returns, periods=252, rf=0.0)
        actual = pm.sortino(qqq_polars, periods_per_year=252, risk_free_rate=0.0)
        np.testing.assert_allclose(actual, expected, **LOOSE)

    def test_calmar_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test Calmar ratio on synthetic data."""
        # Calmar uses CAGR which is calendar-based in QuantStats
        expected = qs.stats.calmar(sample_returns, periods=CALENDAR_PERIODS)
        actual = pm.calmar(polars_returns, periods_per_year=CALENDAR_PERIODS)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_calmar_spy(self, spy_returns: pd.Series, spy_polars: pl.Series) -> None:
        """Test Calmar ratio on SPY data."""
        expected = qs.stats.calmar(spy_returns, periods=CALENDAR_PERIODS)
        actual = pm.calmar(spy_polars, periods_per_year=CALENDAR_PERIODS)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_calmar_qqq(self, qqq_returns: pd.Series, qqq_polars: pl.Series) -> None:
        """Test Calmar ratio on QQQ data."""
        expected = qs.stats.calmar(qqq_returns, periods=CALENDAR_PERIODS)
        actual = pm.calmar(qqq_polars, periods_per_year=CALENDAR_PERIODS)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    def test_omega_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test Omega ratio on synthetic data."""
        expected = qs.stats.omega(sample_returns, rf=0.0, required_return=0.0, periods=CALENDAR_PERIODS)
        actual = pm.omega(polars_returns, threshold=0.0, risk_free_rate=0.0, periods_per_year=CALENDAR_PERIODS)
        np.testing.assert_allclose(actual, expected, **LOOSE)

    @pytest.mark.integration
    def test_omega_spy(self, spy_returns: pd.Series, spy_polars: pl.Series) -> None:
        """Test Omega ratio on SPY data."""
        expected = qs.stats.omega(spy_returns, rf=0.0, required_return=0.0, periods=CALENDAR_PERIODS)
        actual = pm.omega(spy_polars, threshold=0.0, risk_free_rate=0.0, periods_per_year=CALENDAR_PERIODS)
        np.testing.assert_allclose(actual, expected, **LOOSE)

    @pytest.mark.integration
    def test_omega_qqq(self, qqq_returns: pd.Series, qqq_polars: pl.Series) -> None:
        """Test Omega ratio on QQQ data."""
        expected = qs.stats.omega(qqq_returns, rf=0.0, required_return=0.0, periods=CALENDAR_PERIODS)
        actual = pm.omega(qqq_polars, threshold=0.0, risk_free_rate=0.0, periods_per_year=CALENDAR_PERIODS)
        np.testing.assert_allclose(actual, expected, **LOOSE)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_series_comp(self, empty_returns: pl.Series) -> None:
        """Test comp on empty series raises error."""
        with pytest.raises(EmptySeriesError):
            pm.comp(empty_returns)

    def test_empty_series_sharpe(self, empty_returns: pl.Series) -> None:
        """Test sharpe on empty series raises error."""
        with pytest.raises(EmptySeriesError):
            pm.sharpe(empty_returns)

    def test_single_return_comp(self, single_return: pl.Series) -> None:
        """Test comp on single return."""
        result = pm.comp(single_return)
        assert result == 0.05

    def test_all_positive_avg_loss(self, all_positive_returns: pl.Series) -> None:
        """Test avg_loss with no losing days."""
        result = pm.avg_loss(all_positive_returns)
        assert result == 0.0

    def test_all_negative_avg_win(self, all_negative_returns: pl.Series) -> None:
        """Test avg_win with no winning days."""
        result = pm.avg_win(all_negative_returns)
        assert result == 0.0

    def test_flat_returns_volatility(self, flat_returns: pl.Series) -> None:
        """Test volatility on flat returns."""
        result = pm.volatility(flat_returns)
        assert result == 0.0

    def test_flat_returns_sharpe(self, flat_returns: pl.Series) -> None:
        """Test sharpe on flat returns."""
        result = pm.sharpe(flat_returns)
        assert result == 0.0

    def test_max_drawdown_no_drawdown(self, all_positive_returns: pl.Series) -> None:
        """Test max drawdown when always going up."""
        result = pm.max_drawdown(all_positive_returns)
        assert result == 0.0


class TestWinLossMetrics:
    """Test win/loss related metrics."""

    def test_win_rate_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test win rate on synthetic data."""
        expected = qs.stats.win_rate(sample_returns)
        actual = pm.win_rate(polars_returns)
        np.testing.assert_allclose(actual, expected, **EXACT)

    @pytest.mark.integration
    def test_win_rate_spy(self, spy_returns: pd.Series, spy_polars: pl.Series) -> None:
        """Test win rate on SPY data."""
        expected = qs.stats.win_rate(spy_returns)
        actual = pm.win_rate(spy_polars)
        np.testing.assert_allclose(actual, expected, **EXACT)

    def test_payoff_ratio_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test payoff ratio on synthetic data."""
        expected = qs.stats.payoff_ratio(sample_returns)
        actual = pm.payoff_ratio(polars_returns)
        np.testing.assert_allclose(actual, expected, **EXACT)

    @pytest.mark.integration
    def test_payoff_ratio_spy(
        self, spy_returns: pd.Series, spy_polars: pl.Series
    ) -> None:
        """Test payoff ratio on SPY data."""
        expected = qs.stats.payoff_ratio(spy_returns)
        actual = pm.payoff_ratio(spy_polars)
        np.testing.assert_allclose(actual, expected, **EXACT)

    def test_profit_factor_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test profit factor on synthetic data."""
        expected = qs.stats.profit_factor(sample_returns)
        actual = pm.profit_factor(polars_returns)
        np.testing.assert_allclose(actual, expected, **EXACT)

    @pytest.mark.integration
    def test_profit_factor_spy(
        self, spy_returns: pd.Series, spy_polars: pl.Series
    ) -> None:
        """Test profit factor on SPY data."""
        expected = qs.stats.profit_factor(spy_returns)
        actual = pm.profit_factor(spy_polars)
        np.testing.assert_allclose(actual, expected, **EXACT)


class TestAdditionalPerformanceMetrics:
    """Test additional performance metrics."""

    def test_gain_to_pain_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test gain to pain ratio on synthetic data."""
        expected = qs.stats.gain_to_pain_ratio(sample_returns)
        actual = pm.gain_to_pain_ratio(polars_returns)
        np.testing.assert_allclose(actual, expected, **LOOSE)

    @pytest.mark.integration
    def test_gain_to_pain_spy(
        self, spy_returns: pd.Series, spy_polars: pl.Series
    ) -> None:
        """Test gain to pain ratio on SPY data."""
        expected = qs.stats.gain_to_pain_ratio(spy_returns)
        actual = pm.gain_to_pain_ratio(spy_polars)
        np.testing.assert_allclose(actual, expected, **LOOSE)

    def test_tail_ratio_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test tail ratio on synthetic data."""
        expected = qs.stats.tail_ratio(sample_returns)
        actual = pm.tail_ratio(polars_returns)
        np.testing.assert_allclose(actual, expected, **LOOSE)

    @pytest.mark.integration
    def test_tail_ratio_spy(
        self, spy_returns: pd.Series, spy_polars: pl.Series
    ) -> None:
        """Test tail ratio on SPY data."""
        expected = qs.stats.tail_ratio(spy_returns)
        actual = pm.tail_ratio(spy_polars)
        np.testing.assert_allclose(actual, expected, **LOOSE)

    def test_kelly_criterion_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test Kelly criterion on synthetic data."""
        expected = qs.stats.kelly_criterion(sample_returns)
        actual = pm.kelly_criterion(polars_returns)
        np.testing.assert_allclose(actual, expected, **LOOSE)

    @pytest.mark.integration
    def test_kelly_criterion_spy(
        self, spy_returns: pd.Series, spy_polars: pl.Series
    ) -> None:
        """Test Kelly criterion on SPY data."""
        expected = qs.stats.kelly_criterion(spy_returns)
        actual = pm.kelly_criterion(spy_polars)
        np.testing.assert_allclose(actual, expected, **LOOSE)

    def test_recovery_factor_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test recovery factor on synthetic data."""
        expected = qs.stats.recovery_factor(sample_returns)
        actual = pm.recovery_factor(polars_returns)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_recovery_factor_spy(
        self, spy_returns: pd.Series, spy_polars: pl.Series
    ) -> None:
        """Test recovery factor on SPY data."""
        expected = qs.stats.recovery_factor(spy_returns)
        actual = pm.recovery_factor(spy_polars)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    def test_risk_return_ratio_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test risk return ratio on synthetic data."""
        # QuantStats uses mean/std (not annualized)
        expected = qs.stats.risk_return_ratio(sample_returns)
        actual = pm.risk_return_ratio(polars_returns)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_risk_return_ratio_spy(
        self, spy_returns: pd.Series, spy_polars: pl.Series
    ) -> None:
        """Test risk return ratio on SPY data."""
        expected = qs.stats.risk_return_ratio(spy_returns)
        actual = pm.risk_return_ratio(spy_polars)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    def test_common_sense_ratio_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test common sense ratio on synthetic data."""
        expected = qs.stats.common_sense_ratio(sample_returns)
        actual = pm.common_sense_ratio(polars_returns)
        np.testing.assert_allclose(actual, expected, **LOOSE)

    @pytest.mark.integration
    def test_common_sense_ratio_spy(
        self, spy_returns: pd.Series, spy_polars: pl.Series
    ) -> None:
        """Test common sense ratio on SPY data."""
        expected = qs.stats.common_sense_ratio(spy_returns)
        actual = pm.common_sense_ratio(spy_polars)
        np.testing.assert_allclose(actual, expected, **LOOSE)


class TestUlcerMetrics:
    """Test ulcer-related metrics."""

    def test_ulcer_index_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test ulcer index on synthetic data."""
        expected = qs.stats.ulcer_index(sample_returns)
        actual = pm.ulcer_index(polars_returns)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_ulcer_index_spy(
        self, spy_returns: pd.Series, spy_polars: pl.Series
    ) -> None:
        """Test ulcer index on SPY data."""
        expected = qs.stats.ulcer_index(spy_returns)
        actual = pm.ulcer_index(spy_polars)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    def test_ulcer_performance_index_synthetic(
        self, sample_returns: pd.Series, polars_returns: pl.Series
    ) -> None:
        """Test ulcer performance index on synthetic data."""
        expected = qs.stats.ulcer_performance_index(sample_returns, rf=0.0)
        actual = pm.ulcer_performance_index(polars_returns, risk_free_rate=0.0)
        np.testing.assert_allclose(actual, expected, **TIGHT)

    @pytest.mark.integration
    def test_ulcer_performance_index_spy(
        self, spy_returns: pd.Series, spy_polars: pl.Series
    ) -> None:
        """Test ulcer performance index on SPY data."""
        expected = qs.stats.ulcer_performance_index(spy_returns, rf=0.0)
        actual = pm.ulcer_performance_index(spy_polars, risk_free_rate=0.0)
        np.testing.assert_allclose(actual, expected, **TIGHT)
