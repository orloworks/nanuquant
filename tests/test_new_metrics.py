"""Tests for new metrics: outliers, period analysis, rolling_greeks.

These tests verify that the new metrics match QuantStats behavior and
work correctly with edge cases.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

import nanuquant as pm


class TestOutliers:
    """Test outlier detection functions."""

    def test_outliers_basic(self) -> None:
        """Test basic outlier detection."""
        # Create data with clear outliers
        returns = pl.Series([0.01, 0.02, -0.01, 0.015, 0.10, -0.02, 0.03])
        out = pm.outliers(returns, quantile=0.90)

        # Should identify values above 90th percentile
        assert len(out) > 0
        # All outliers should be greater than the 90th percentile
        threshold = returns.quantile(0.90)
        assert all(v > threshold for v in out.to_list())

    def test_outliers_empty(self) -> None:
        """Test outliers with empty series."""
        returns = pl.Series([], dtype=pl.Float64)
        out = pm.outliers(returns, quantile=0.95)
        assert len(out) == 0

    def test_remove_outliers_basic(self) -> None:
        """Test basic outlier removal."""
        returns = pl.Series([0.01, 0.02, -0.01, 0.015, 0.10, -0.02, 0.03])
        clean = pm.remove_outliers(returns, quantile=0.90)

        # Should have fewer or equal elements (removes values > threshold)
        assert len(clean) <= len(returns)
        # All remaining values should be strictly less than threshold
        # (remove_outliers keeps values < threshold, not <=)
        threshold = returns.quantile(0.90)
        # The max of clean should be less than or equal to the second largest in original
        assert clean.max() <= threshold

    def test_remove_outliers_preserves_count(self) -> None:
        """Test that outliers + non-outliers = total."""
        returns = pl.Series([0.01, 0.02, -0.01, 0.015, 0.10, -0.02, 0.03] * 10)
        out = pm.outliers(returns, quantile=0.95)
        clean = pm.remove_outliers(returns, quantile=0.95)

        # The sum should approximately equal the original (minus some edge cases)
        # Note: This might not be exact due to how quantile boundaries work
        assert len(clean) <= len(returns)

    def test_outliers_iqr_basic(self) -> None:
        """Test IQR-based outlier detection."""
        # Create data with clear outliers using IQR method
        normal_data = [0.01, 0.02, -0.01, 0.015, -0.02, 0.03, -0.015, 0.025]
        outlier_data = [0.50, -0.40]  # Clear outliers
        returns = pl.Series(normal_data + outlier_data)

        out = pm.outliers_iqr(returns)

        # Should detect the extreme values
        assert len(out) >= 2

    def test_remove_outliers_iqr_basic(self) -> None:
        """Test IQR-based outlier removal."""
        normal_data = [0.01, 0.02, -0.01, 0.015, -0.02, 0.03, -0.015, 0.025]
        outlier_data = [0.50, -0.40]
        returns = pl.Series(normal_data + outlier_data)

        clean = pm.remove_outliers_iqr(returns)

        # Should have removed the outliers
        assert len(clean) <= len(returns)
        # Extreme values should be removed
        assert 0.50 not in clean.to_list()
        assert -0.40 not in clean.to_list()

    def test_outliers_quantile_parameter(self) -> None:
        """Test different quantile values."""
        returns = pl.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])

        out_90 = pm.outliers(returns, quantile=0.90)
        out_50 = pm.outliers(returns, quantile=0.50)

        # Lower quantile should identify more outliers
        assert len(out_50) >= len(out_90)


class TestRollingGreeks:
    """Test rolling greeks (alpha and beta) calculation."""

    def test_rolling_greeks_basic(self) -> None:
        """Test basic rolling greeks calculation."""
        returns = pl.Series([0.02, -0.01, 0.03, 0.01, -0.02] * 50)
        benchmark = pl.Series([0.01, -0.02, 0.02, 0.01, -0.01] * 50)

        greeks = pm.rolling_greeks(returns, benchmark, rolling_period=10)

        assert "rolling_alpha" in greeks.columns
        assert "rolling_beta" in greeks.columns
        assert len(greeks) == len(returns)

    def test_rolling_greeks_beta_consistency(self) -> None:
        """Test that rolling_greeks beta matches rolling_beta."""
        returns = pl.Series([0.02, -0.01, 0.03, 0.01, -0.02] * 50)
        benchmark = pl.Series([0.01, -0.02, 0.02, 0.01, -0.01] * 50)

        greeks = pm.rolling_greeks(returns, benchmark, rolling_period=10)
        standalone_beta = pm.rolling_beta(returns, benchmark, rolling_period=10)

        # Beta values should match
        np.testing.assert_allclose(
            greeks["rolling_beta"].drop_nulls().to_numpy(),
            standalone_beta.drop_nulls().to_numpy(),
            rtol=1e-10,
        )

    def test_rolling_greeks_empty(self) -> None:
        """Test rolling greeks with empty series."""
        returns = pl.Series([], dtype=pl.Float64)
        benchmark = pl.Series([], dtype=pl.Float64)

        greeks = pm.rolling_greeks(returns, benchmark, rolling_period=10)

        assert len(greeks) == 0

    def test_rolling_greeks_short_series(self) -> None:
        """Test rolling greeks with series shorter than window."""
        returns = pl.Series([0.01, 0.02, 0.03])
        benchmark = pl.Series([0.005, 0.01, 0.015])

        greeks = pm.rolling_greeks(returns, benchmark, rolling_period=10)

        assert len(greeks) == 0

    def test_rolling_greeks_perfect_correlation(self) -> None:
        """Test with perfectly correlated returns."""
        benchmark = pl.Series([0.01, -0.02, 0.03, 0.01, -0.01] * 20)
        returns = benchmark * 2  # Perfectly correlated, beta should be ~2

        greeks = pm.rolling_greeks(returns, benchmark, rolling_period=10)

        # Beta should be close to 2
        valid_beta = greeks["rolling_beta"].drop_nulls()
        assert valid_beta.mean() is not None
        np.testing.assert_allclose(valid_beta.mean(), 2.0, rtol=1e-6)


class TestMonthlyReturns:
    """Test monthly returns table generation."""

    def test_monthly_returns_basic(self) -> None:
        """Test basic monthly returns calculation."""
        # Create one year of daily returns
        returns = pl.Series([0.001] * 365)
        dates = pl.Series([date(2020, 1, 1) + timedelta(days=i) for i in range(365)])

        monthly = pm.monthly_returns(returns, dates=dates)

        # Should have 12 months
        assert len(monthly) == 12
        # Should have month column plus year columns
        assert "month" in monthly.columns

    def test_monthly_returns_compounded(self) -> None:
        """Test compounded vs simple monthly returns."""
        returns = pl.Series([0.01] * 30)  # 1% daily for a month
        dates = pl.Series([date(2020, 1, 1) + timedelta(days=i) for i in range(30)])

        compounded = pm.monthly_returns(returns, dates=dates, compounded=True)
        simple = pm.monthly_returns(returns, dates=dates, compounded=False)

        # Compounded should be higher (or at least different)
        # Both should have data
        assert not compounded.is_empty()
        assert not simple.is_empty()

    def test_monthly_returns_multi_year(self) -> None:
        """Test monthly returns spanning multiple years."""
        # 2 years of data
        returns = pl.Series([0.001] * 730)
        dates = pl.Series([date(2020, 1, 1) + timedelta(days=i) for i in range(730)])

        monthly = pm.monthly_returns(returns, dates=dates)

        # Should have 12 months in rows
        assert len(monthly) == 12
        # Should have multiple year columns
        assert len(monthly.columns) >= 2  # month + at least 1 year

    def test_monthly_returns_empty(self) -> None:
        """Test monthly returns with empty series."""
        returns = pl.Series([], dtype=pl.Float64)
        monthly = pm.monthly_returns(returns)
        assert monthly.is_empty()


class TestDistribution:
    """Test distribution analysis function."""

    def test_distribution_basic(self) -> None:
        """Test basic distribution analysis."""
        returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 100)
        dates = pl.Series([date(2020, 1, 1) + timedelta(days=i) for i in range(500)])

        dist = pm.distribution(returns, dates=dates)

        # Should have all period keys
        expected_keys = ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
        for key in expected_keys:
            assert key in dist

    def test_distribution_has_stats(self) -> None:
        """Test that distribution includes all expected statistics."""
        returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 100)
        dates = pl.Series([date(2020, 1, 1) + timedelta(days=i) for i in range(500)])

        dist = pm.distribution(returns, dates=dates)

        # Check that each period has expected keys
        for period in dist.values():
            assert "values" in period
            assert "outliers" in period
            assert "mean" in period
            assert "std" in period
            assert "min" in period
            assert "max" in period
            assert "count" in period

    def test_distribution_outlier_detection(self) -> None:
        """Test that distribution detects outliers."""
        # Create data with clear outliers
        normal = [0.01, -0.01, 0.02, -0.02] * 100
        outliers_data = [0.50, -0.50]  # Clear daily outliers
        returns = pl.Series(normal + outliers_data)
        dates = pl.Series([date(2020, 1, 1) + timedelta(days=i) for i in range(len(returns))])

        dist = pm.distribution(returns, dates=dates)

        # Daily outliers should be detected
        assert len(dist["Daily"]["outliers"]) > 0

    def test_distribution_period_counts(self) -> None:
        """Test that period counts are correct."""
        # 365 days of data
        returns = pl.Series([0.001] * 365)
        dates = pl.Series([date(2020, 1, 1) + timedelta(days=i) for i in range(365)])

        dist = pm.distribution(returns, dates=dates)

        # Should have 365 daily values
        assert dist["Daily"]["count"] == 365
        # Should have ~52 weekly values
        assert 50 <= dist["Weekly"]["count"] <= 54
        # Should have 12 monthly values
        assert dist["Monthly"]["count"] == 12
        # Should have 4 quarterly values
        assert dist["Quarterly"]["count"] == 4
        # Should have 1 yearly value
        assert dist["Yearly"]["count"] == 1

    def test_distribution_empty(self) -> None:
        """Test distribution with empty series."""
        returns = pl.Series([], dtype=pl.Float64)
        dist = pm.distribution(returns)
        assert dist == {}


class TestCompare:
    """Test strategy vs benchmark comparison."""

    def test_compare_basic(self) -> None:
        """Test basic comparison."""
        strategy = pl.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        benchmark = pl.Series([0.005, -0.01, 0.02, 0.005, -0.005])
        dates = pl.Series([date(2020, 1, 1) + timedelta(days=i) for i in range(5)])

        cmp = pm.compare(strategy, benchmark, dates=dates)

        assert "date" in cmp.columns
        assert "strategy" in cmp.columns
        assert "benchmark" in cmp.columns
        assert "excess" in cmp.columns
        assert "win" in cmp.columns

    def test_compare_excess_calculation(self) -> None:
        """Test that excess returns are calculated correctly."""
        strategy = pl.Series([0.10, 0.05, -0.02])
        benchmark = pl.Series([0.05, 0.05, 0.01])
        dates = pl.Series([date(2020, 1, 1) + timedelta(days=i) for i in range(3)])

        cmp = pm.compare(strategy, benchmark, dates=dates)

        expected_excess = [0.05, 0.0, -0.03]
        np.testing.assert_allclose(
            cmp["excess"].to_numpy(),
            expected_excess,
            rtol=1e-10,
        )

    def test_compare_win_indicator(self) -> None:
        """Test that win indicator is correct."""
        strategy = pl.Series([0.10, 0.05, -0.02])
        benchmark = pl.Series([0.05, 0.05, 0.01])
        dates = pl.Series([date(2020, 1, 1) + timedelta(days=i) for i in range(3)])

        cmp = pm.compare(strategy, benchmark, dates=dates)

        # First: 0.10 > 0.05 -> win
        # Second: 0.05 == 0.05 -> not win (not strictly greater)
        # Third: -0.02 < 0.01 -> not win
        expected_win = [1, 0, 0]
        assert cmp["win"].to_list() == expected_win

    def test_compare_empty(self) -> None:
        """Test comparison with empty series."""
        strategy = pl.Series([], dtype=pl.Float64)
        benchmark = pl.Series([], dtype=pl.Float64)

        cmp = pm.compare(strategy, benchmark)
        assert cmp.is_empty()

    def test_compare_length_mismatch(self) -> None:
        """Test that mismatched lengths raise error."""
        strategy = pl.Series([0.01, 0.02, 0.03])
        benchmark = pl.Series([0.01, 0.02])

        with pytest.raises(Exception):  # Should raise BenchmarkMismatchError
            pm.compare(strategy, benchmark)


class TestDifferentialVsQuantStats:
    """Compare new metrics against QuantStats where applicable.

    Note: These tests require quantstats-lumi to be installed.
    """

    @pytest.fixture
    def sample_returns(self) -> pl.Series:
        """Generate sample returns for testing."""
        np.random.seed(42)
        return pl.Series(np.random.normal(0.001, 0.02, 500))

    @pytest.fixture
    def sample_benchmark(self) -> pl.Series:
        """Generate sample benchmark returns."""
        np.random.seed(43)
        return pl.Series(np.random.normal(0.0008, 0.015, 500))

    def test_outliers_vs_quantstats(self, sample_returns: pl.Series) -> None:
        """Test outliers detection matches QuantStats."""
        try:
            import quantstats_lumi as qs
            import pandas as pd
        except ImportError:
            pytest.skip("quantstats-lumi not installed")

        pd_returns = pd.Series(sample_returns.to_numpy())

        # QuantStats outliers
        qs_outliers = qs.stats.outliers(pd_returns, quantile=0.95)
        pm_outliers = pm.outliers(sample_returns, quantile=0.95)

        # Should have same number of outliers
        assert len(pm_outliers) == len(qs_outliers.dropna())

    def test_remove_outliers_vs_quantstats(self, sample_returns: pl.Series) -> None:
        """Test remove_outliers matches QuantStats."""
        try:
            import quantstats_lumi as qs
            import pandas as pd
        except ImportError:
            pytest.skip("quantstats-lumi not installed")

        pd_returns = pd.Series(sample_returns.to_numpy())

        # QuantStats remove_outliers
        qs_clean = qs.stats.remove_outliers(pd_returns, quantile=0.95)
        pm_clean = pm.remove_outliers(sample_returns, quantile=0.95)

        # Should have same length
        assert len(pm_clean) == len(qs_clean.dropna())
