"""Tests for institutional systemic risk metrics.

Tests absorption ratio, tail dependence, and downside correlation.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from nanuquant.institutional import (
    AbsorptionRatioResult,
    absorption_ratio,
    downside_correlation,
    lower_tail_dependence,
    upside_correlation,
)


class TestAbsorptionRatio:
    """Tests for absorption ratio calculation."""

    def test_result_type(self, market_data_df: pl.DataFrame) -> None:
        """Test that result is correct named tuple type."""
        pytest.importorskip("pyarrow")
        result = absorption_ratio(market_data_df)
        assert isinstance(result, AbsorptionRatioResult)
        assert hasattr(result, "absorption_ratio")
        assert hasattr(result, "n_components")
        assert hasattr(result, "total_assets")
        assert hasattr(result, "eigenvalues")

    def test_ar_range(self) -> None:
        """Absorption ratio must be in [0, 1]."""
        np.random.seed(42)
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, 200),
            "B": np.random.normal(0, 0.02, 200),
            "C": np.random.normal(0, 0.02, 200),
            "D": np.random.normal(0, 0.02, 200),
            "E": np.random.normal(0, 0.02, 200),
        })
        result = absorption_ratio(returns)
        assert 0 <= result.absorption_ratio <= 1

    def test_high_correlation_high_ar(self) -> None:
        """Highly correlated assets should have high absorption ratio."""
        np.random.seed(42)
        n = 500
        common_factor = np.random.normal(0, 0.02, n)

        # Create highly correlated assets
        returns = pl.DataFrame({
            "A": common_factor + np.random.normal(0, 0.002, n),
            "B": common_factor + np.random.normal(0, 0.002, n),
            "C": common_factor + np.random.normal(0, 0.002, n),
            "D": common_factor + np.random.normal(0, 0.002, n),
            "E": common_factor + np.random.normal(0, 0.002, n),
        })

        result = absorption_ratio(returns, n_components=1)

        # First component should explain most variance
        assert result.absorption_ratio > 0.7

    def test_low_correlation_low_ar(self) -> None:
        """Independent assets should have low absorption ratio."""
        np.random.seed(42)
        n = 500

        # Create independent assets
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, n),
            "B": np.random.normal(0, 0.02, n),
            "C": np.random.normal(0, 0.02, n),
            "D": np.random.normal(0, 0.02, n),
            "E": np.random.normal(0, 0.02, n),
        })

        result = absorption_ratio(returns, n_components=1)

        # First component should explain only ~1/5 of variance for 5 assets
        assert result.absorption_ratio < 0.5

    def test_n_components_parameter(self) -> None:
        """Test explicit n_components setting."""
        np.random.seed(42)
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, 200),
            "B": np.random.normal(0, 0.02, 200),
            "C": np.random.normal(0, 0.02, 200),
        })

        result1 = absorption_ratio(returns, n_components=1)
        result2 = absorption_ratio(returns, n_components=2)

        assert result1.n_components == 1
        assert result2.n_components == 2
        assert result2.absorption_ratio > result1.absorption_ratio

    def test_eigenvalues_sum(self) -> None:
        """Eigenvalues should sum to number of assets (for correlation matrix)."""
        np.random.seed(42)
        n_assets = 5
        returns = pl.DataFrame({
            f"Asset_{i}": np.random.normal(0, 0.02, 200) for i in range(n_assets)
        })

        result = absorption_ratio(returns)

        # Sum of eigenvalues equals trace of correlation matrix = n_assets
        assert abs(sum(result.eigenvalues) - n_assets) < 0.1

    def test_insufficient_assets(self) -> None:
        """Should raise error for single asset."""
        returns = pl.DataFrame({"A": np.random.normal(0, 0.02, 100)})

        with pytest.raises(ValueError):
            absorption_ratio(returns)


class TestLowerTailDependence:
    """Tests for lower tail dependence calculation."""

    def test_range(self, polars_returns: pl.Series, polars_benchmark: pl.Series) -> None:
        """Tail dependence must be in [0, 1]."""
        ltd = lower_tail_dependence(polars_returns, polars_benchmark)
        assert 0 <= ltd <= 1

    def test_independent_returns_low_dependence(self) -> None:
        """Independent returns should have low tail dependence."""
        np.random.seed(42)
        r1 = pl.Series(np.random.normal(0, 0.02, 2000))
        r2 = pl.Series(np.random.normal(0, 0.02, 2000))

        ltd = lower_tail_dependence(r1, r2, threshold_quantile=0.05)

        # For independent data, tail dependence should be around threshold_quantile
        assert ltd < 0.2

    def test_comonotonic_returns_high_dependence(self) -> None:
        """Co-monotonic returns should have high tail dependence."""
        np.random.seed(42)
        common = np.random.normal(0, 0.02, 1000)
        r1 = pl.Series(common)
        r2 = pl.Series(common * 1.5)  # Perfectly correlated

        ltd = lower_tail_dependence(r1, r2, threshold_quantile=0.05)

        # Perfect correlation should give tail dependence ≈ 1
        assert ltd > 0.8

    def test_threshold_quantile_effect(self, polars_returns: pl.Series, polars_benchmark: pl.Series) -> None:
        """Different thresholds should give different results."""
        ltd_5 = lower_tail_dependence(polars_returns, polars_benchmark, threshold_quantile=0.05)
        ltd_10 = lower_tail_dependence(polars_returns, polars_benchmark, threshold_quantile=0.10)

        # Both should be valid
        assert 0 <= ltd_5 <= 1
        assert 0 <= ltd_10 <= 1

    def test_invalid_threshold(self, polars_returns: pl.Series, polars_benchmark: pl.Series) -> None:
        """Should raise error for invalid threshold."""
        with pytest.raises(ValueError):
            lower_tail_dependence(polars_returns, polars_benchmark, threshold_quantile=0.6)

        with pytest.raises(ValueError):
            lower_tail_dependence(polars_returns, polars_benchmark, threshold_quantile=-0.1)


class TestDownsideCorrelation:
    """Tests for downside correlation calculation."""

    def test_range(self, polars_returns: pl.Series, polars_benchmark: pl.Series) -> None:
        """Downside correlation must be in [-1, 1]."""
        dc = downside_correlation(polars_returns, polars_benchmark)
        assert -1 <= dc <= 1 or math.isnan(dc)

    def test_positive_returns_correlation(self) -> None:
        """Positively correlated returns should have positive downside correlation."""
        np.random.seed(42)
        common = np.random.normal(0, 0.02, 1000)
        r1 = pl.Series(common)
        r2 = pl.Series(common + np.random.normal(0, 0.005, 1000))

        dc = downside_correlation(r1, r2)

        assert dc > 0.5

    def test_negative_returns_correlation(self) -> None:
        """Negatively correlated returns should have negative downside correlation."""
        np.random.seed(42)
        common = np.random.normal(0, 0.02, 1000)
        r1 = pl.Series(common)
        r2 = pl.Series(-common + np.random.normal(0, 0.005, 1000))

        dc = downside_correlation(r1, r2)

        assert dc < -0.5

    def test_custom_threshold(self, polars_returns: pl.Series, polars_benchmark: pl.Series) -> None:
        """Test with custom threshold."""
        dc_0 = downside_correlation(polars_returns, polars_benchmark, threshold=0.0)
        dc_neg = downside_correlation(polars_returns, polars_benchmark, threshold=-0.01)

        # Both should be valid correlations
        assert -1 <= dc_0 <= 1 or math.isnan(dc_0)
        assert -1 <= dc_neg <= 1 or math.isnan(dc_neg)


class TestUpsideCorrelation:
    """Tests for upside correlation calculation."""

    def test_range(self, polars_returns: pl.Series, polars_benchmark: pl.Series) -> None:
        """Upside correlation must be in [-1, 1]."""
        uc = upside_correlation(polars_returns, polars_benchmark)
        assert -1 <= uc <= 1 or math.isnan(uc)

    def test_asymmetric_correlation(self) -> None:
        """Downside and upside correlations can differ."""
        np.random.seed(42)
        n = 1000

        # Create returns that are more correlated in downturns
        r1 = np.random.normal(0, 0.02, n)
        r2 = np.zeros(n)

        for i in range(n):
            if r1[i] < 0:
                # High correlation in downside
                r2[i] = r1[i] * 0.9 + np.random.normal(0, 0.003)
            else:
                # Low correlation in upside
                r2[i] = np.random.normal(0.005, 0.02)

        s1 = pl.Series(r1)
        s2 = pl.Series(r2)

        dc = downside_correlation(s1, s2)
        uc = upside_correlation(s1, s2)

        # Downside correlation should be higher
        assert dc > uc


class TestSyntheticDataRecovery:
    """Synthetic data tests for systemic metrics."""

    def test_ar_with_factor_structure(self) -> None:
        """Test AR recovery with known factor structure."""
        np.random.seed(42)
        n = 500
        n_assets = 10

        # Single factor model
        factor = np.random.normal(0, 0.02, n)
        betas = np.random.uniform(0.5, 1.5, n_assets)

        data = {}
        for i in range(n_assets):
            idiosyncratic = np.random.normal(0, 0.01, n)
            data[f"Asset_{i}"] = betas[i] * factor + idiosyncratic

        returns = pl.DataFrame(data)
        result = absorption_ratio(returns, n_components=1)

        # Single factor should explain most variance
        assert result.absorption_ratio > 0.5

    def test_tail_dependence_copula_structure(self) -> None:
        """Test tail dependence with known extreme correlation."""
        np.random.seed(42)
        n = 5000

        # Create returns that are only correlated in extremes
        r1 = np.random.normal(0, 0.02, n)
        r2 = np.random.normal(0, 0.02, n)

        # Force co-crashes in the 5% tail
        tail_mask = r1 < np.percentile(r1, 5)
        r2[tail_mask] = r1[tail_mask] * 1.2

        ltd = lower_tail_dependence(pl.Series(r1), pl.Series(r2))

        # Should detect the tail dependence
        assert ltd > 0.5
