"""Tests for nanuquant.core.timeseries module.

Tests the new timeseries functions that return arrays/DataFrames:
- yearly_returns
- drawdown_details
- histogram
- cumulative_returns (alias)
- equity_curve (alias)
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

import nanuquant as pm
from nanuquant.core.timeseries import (
    cumulative_returns,
    drawdown_details,
    equity_curve,
    histogram,
    yearly_returns,
)


class TestYearlyReturns:
    """Tests for yearly_returns function."""

    def test_basic_yearly_returns(self):
        """Test basic yearly returns calculation."""
        # Create 3 years of daily data
        n_days = 252 * 3
        dates = pl.Series(
            "date",
            [date(2020, 1, 1) + timedelta(days=i) for i in range(n_days)],
        )
        returns = pl.Series("returns", [0.001] * n_days)

        result = yearly_returns(returns, dates=dates)

        assert isinstance(result, pl.DataFrame)
        assert "year" in result.columns
        assert "return" in result.columns
        assert len(result) == 3  # 2020, 2021, 2022

    def test_compounded_vs_simple(self):
        """Test compounded vs simple aggregation."""
        dates = pl.Series(
            "date",
            [date(2020, 1, 1) + timedelta(days=i) for i in range(10)],
        )
        returns = pl.Series("returns", [0.01] * 10)

        compounded = yearly_returns(returns, dates=dates, compounded=True)
        simple = yearly_returns(returns, dates=dates, compounded=False)

        # Compounded should be higher than simple sum
        assert compounded["return"][0] > simple["return"][0]
        # Simple should just be sum
        assert abs(simple["return"][0] - 0.10) < 1e-10

    def test_empty_returns(self):
        """Test with empty returns series."""
        empty = pl.Series("returns", [], dtype=pl.Float64)
        result = yearly_returns(empty)
        assert result.is_empty()

    def test_sorted_by_year(self):
        """Test that results are sorted by year."""
        # Create data spanning multiple years in random order is handled
        dates = pl.Series(
            "date",
            [date(2022, 6, 1), date(2020, 1, 1), date(2021, 3, 15)],
        )
        returns = pl.Series("returns", [0.05, 0.03, 0.04])

        result = yearly_returns(returns, dates=dates)
        years = result["year"].to_list()
        assert years == sorted(years)

    def test_top_level_export(self):
        """Test that yearly_returns is exported at top level."""
        assert hasattr(pm, "yearly_returns")
        assert pm.yearly_returns is yearly_returns


class TestDrawdownDetails:
    """Tests for drawdown_details function."""

    def test_basic_drawdown_detection(self):
        """Test basic drawdown period detection."""
        # Create returns with a clear drawdown
        returns = pl.Series([0.10, -0.15, -0.05, 0.20, 0.10])

        result = drawdown_details(returns, top_n=5)

        assert isinstance(result, pl.DataFrame)
        assert "start" in result.columns
        assert "valley" in result.columns
        assert "end" in result.columns
        assert "depth" in result.columns
        assert "length" in result.columns
        assert "recovery" in result.columns

    def test_multiple_drawdowns(self):
        """Test detection of multiple drawdown periods."""
        # Create returns with two distinct drawdowns
        returns = pl.Series([
            0.10, -0.20, 0.25,  # First drawdown and recovery
            0.05, -0.15, -0.05, 0.30,  # Second drawdown and recovery
        ])

        result = drawdown_details(returns, top_n=5)
        assert len(result) >= 1  # At least one drawdown detected

    def test_top_n_limit(self):
        """Test that top_n limits results."""
        # Create many small drawdowns
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0, 0.02, 500))

        result_5 = drawdown_details(returns, top_n=5)
        result_3 = drawdown_details(returns, top_n=3)

        assert len(result_5) <= 5
        assert len(result_3) <= 3
        assert len(result_3) <= len(result_5)

    def test_sorted_by_depth(self):
        """Test that results are sorted by depth (worst first)."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0.001, 0.02, 500))

        result = drawdown_details(returns, top_n=5)

        if len(result) > 1:
            depths = result["depth"].to_list()
            assert depths == sorted(depths)  # Most negative first

    def test_with_dates(self):
        """Test drawdown details with date series."""
        dates = pl.Series(
            "date",
            [date(2020, 1, i) for i in range(1, 8)],
        )
        returns = pl.Series([0.10, -0.20, -0.05, 0.15, 0.10, -0.08, 0.05])

        result = drawdown_details(returns, dates=dates, top_n=3)

        # Dates should be used in output
        assert result["start"].dtype == pl.Date

    def test_empty_returns(self):
        """Test with empty returns."""
        empty = pl.Series([], dtype=pl.Float64)
        result = drawdown_details(empty)
        assert result.is_empty()

    def test_all_positive_returns(self, all_positive_returns):
        """Test with all positive returns (no drawdowns)."""
        result = drawdown_details(all_positive_returns)
        assert result.is_empty()

    def test_top_level_export(self):
        """Test that drawdown_details is exported at top level."""
        assert hasattr(pm, "drawdown_details")
        assert pm.drawdown_details is drawdown_details


class TestHistogram:
    """Tests for histogram function."""

    def test_basic_histogram(self):
        """Test basic histogram calculation."""
        returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 100)

        result = histogram(returns, bins=20)

        assert isinstance(result, pl.DataFrame)
        assert "bin_start" in result.columns
        assert "bin_end" in result.columns
        assert "bin_center" in result.columns
        assert "count" in result.columns
        assert "frequency" in result.columns
        assert len(result) == 20

    def test_bin_count_sums_to_total(self):
        """Test that bin counts sum to total observations."""
        returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 100)

        result = histogram(returns, bins=20)
        total_count = result["count"].sum()

        assert total_count == len(returns)

    def test_frequencies_sum_to_one(self):
        """Test that frequencies sum to 1."""
        returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 100)

        result = histogram(returns, bins=20)
        freq_sum = result["frequency"].sum()

        assert abs(freq_sum - 1.0) < 1e-10

    def test_density_mode(self):
        """Test density calculation."""
        returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 100)

        result = histogram(returns, bins=20, density=True)

        assert "density" in result.columns
        # Density * bin_width should sum to ~1
        bin_width = result["bin_end"][0] - result["bin_start"][0]
        density_integral = (result["density"] * bin_width).sum()
        assert abs(density_integral - 1.0) < 0.1  # Allow some tolerance

    def test_different_bin_counts(self):
        """Test with different bin counts."""
        returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02] * 100)

        for bins in [10, 25, 50, 100]:
            result = histogram(returns, bins=bins)
            assert len(result) == bins

    def test_empty_returns(self):
        """Test with empty returns."""
        empty = pl.Series([], dtype=pl.Float64)
        result = histogram(empty)
        assert result.is_empty()

    def test_single_value(self):
        """Test with single value (edge case)."""
        returns = pl.Series([0.05])
        result = histogram(returns, bins=10)
        assert result["count"].sum() == 1

    def test_constant_values(self):
        """Test with all same values."""
        returns = pl.Series([0.01] * 100)
        result = histogram(returns, bins=10)
        assert result["count"].sum() == 100

    def test_top_level_export(self):
        """Test that histogram is exported at top level."""
        assert hasattr(pm, "histogram")
        assert pm.histogram is histogram


class TestCumulativeReturnsAlias:
    """Tests for cumulative_returns alias."""

    def test_matches_compound_returns(self, polars_returns):
        """Test that cumulative_returns matches compound_returns."""
        from nanuquant.core.utils import compound_returns as original

        result_alias = cumulative_returns(polars_returns)
        result_original = original(polars_returns)

        assert result_alias.equals(result_original)

    def test_basic_calculation(self):
        """Test basic cumulative return calculation."""
        returns = pl.Series([0.01, 0.02, -0.01, 0.03])

        result = cumulative_returns(returns)

        assert len(result) == len(returns)
        # First value should equal first return
        assert abs(result[0] - 0.01) < 1e-10
        # Should be cumulative product minus 1
        expected_final = (1.01 * 1.02 * 0.99 * 1.03) - 1
        assert abs(result[-1] - expected_final) < 1e-10

    def test_top_level_export(self):
        """Test that cumulative_returns is exported at top level."""
        assert hasattr(pm, "cumulative_returns")
        assert pm.cumulative_returns is cumulative_returns


class TestEquityCurveAlias:
    """Tests for equity_curve alias."""

    def test_matches_compound_returns(self, polars_returns):
        """Test that equity_curve matches compound_returns."""
        from nanuquant.core.utils import compound_returns as original

        result_alias = equity_curve(polars_returns)
        result_original = original(polars_returns)

        assert result_alias.equals(result_original)

    def test_basic_calculation(self):
        """Test basic equity curve calculation."""
        returns = pl.Series([0.10, -0.05, 0.08])

        result = equity_curve(returns)

        assert len(result) == len(returns)
        # Growth of $1: $1 -> $1.10 -> $1.045 -> $1.1286
        expected = [(1.10) - 1, (1.10 * 0.95) - 1, (1.10 * 0.95 * 1.08) - 1]
        for i, exp in enumerate(expected):
            assert abs(result[i] - exp) < 1e-10

    def test_top_level_export(self):
        """Test that equity_curve is exported at top level."""
        assert hasattr(pm, "equity_curve")
        assert pm.equity_curve is equity_curve


class TestLogReturnsExport:
    """Tests for log_returns top-level export."""

    def test_log_returns_exported(self):
        """Test that log_returns is exported at top level."""
        assert hasattr(pm, "log_returns")

    def test_log_returns_calculation(self):
        """Test log_returns calculation."""
        returns = pl.Series([0.01, 0.02, -0.01])

        result = pm.log_returns(returns)

        # log(1 + r)
        expected = (1 + returns).log()
        assert result.equals(expected)

    def test_log_returns_additive(self):
        """Test that log returns are additive (key mathematical property)."""
        # Simple returns: (1+r1)(1+r2) - 1
        # Log returns: log(1+r1) + log(1+r2)
        r1 = pl.Series([0.05])
        r2 = pl.Series([0.03])

        log_r1 = pm.log_returns(r1)
        log_r2 = pm.log_returns(r2)

        # Sum of log returns
        log_sum = log_r1[0] + log_r2[0]

        # Compound simple return -> log
        compound_simple = (1 + r1[0]) * (1 + r2[0]) - 1
        log_compound = pm.log_returns(pl.Series([compound_simple]))[0]

        assert abs(log_sum - log_compound) < 1e-10


class TestSimpleReturnsExport:
    """Tests for simple_returns top-level export."""

    def test_simple_returns_exported(self):
        """Test that simple_returns is exported at top level."""
        assert hasattr(pm, "simple_returns")

    def test_roundtrip_conversion(self):
        """Test roundtrip: simple -> log -> simple."""
        returns = pl.Series([0.01, 0.02, -0.01, 0.03])

        log_rets = pm.log_returns(returns)
        simple_rets = pm.simple_returns(log_rets)

        # Should recover original
        for i in range(len(returns)):
            assert abs(simple_rets[i] - returns[i]) < 1e-10


class TestCompoundReturnsExport:
    """Tests for compound_returns top-level export."""

    def test_compound_returns_exported(self):
        """Test that compound_returns is exported at top level."""
        assert hasattr(pm, "compound_returns")

    def test_compound_returns_calculation(self):
        """Test compound_returns calculation."""
        returns = pl.Series([0.01, 0.02, -0.01])

        result = pm.compound_returns(returns)

        # (1 + r).cumprod() - 1
        expected = (1 + returns).cum_prod() - 1
        assert result.equals(expected)


class TestIntegrationWithMarketData:
    """Integration tests with real market data."""

    def test_yearly_returns_with_spy(self, spy_returns, spy_polars):
        """Test yearly returns with SPY data."""
        import pandas as pd

        # Get dates from spy_returns
        dates = pl.Series("date", spy_returns.index.to_list())
        returns = spy_polars

        result = yearly_returns(returns, dates=dates)

        assert len(result) >= 3  # At least 3 years
        # All returns should be reasonable (-50% to +100%)
        for ret in result["return"].to_list():
            assert -0.5 < ret < 1.0

    def test_drawdown_details_with_spy(self, spy_polars):
        """Test drawdown details with SPY data."""
        result = drawdown_details(spy_polars, top_n=5)

        assert len(result) == 5
        # All depths should be negative
        for depth in result["depth"].to_list():
            assert depth < 0

    def test_histogram_with_spy(self, spy_polars):
        """Test histogram with SPY data."""
        result = histogram(spy_polars, bins=50)

        assert len(result) == 50
        assert result["count"].sum() == len(spy_polars)
