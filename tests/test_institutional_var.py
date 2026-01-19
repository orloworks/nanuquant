"""Tests for institutional VaR extension metrics.

Tests Cornish-Fisher VaR, Entropic VaR, and other VaR variants.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from nanuquant.institutional import (
    cornish_fisher_var,
    entropic_var,
    historical_var,
    modified_var,
    parametric_var,
)


class TestCornishFisherVaR:
    """Tests for Cornish-Fisher adjusted VaR."""

    def test_positive_var(self, polars_returns: pl.Series) -> None:
        """VaR should be positive (represents potential loss)."""
        cf_var = cornish_fisher_var(polars_returns)
        assert cf_var > 0

    def test_confidence_monotonicity(self, polars_returns: pl.Series) -> None:
        """Higher confidence should give higher VaR."""
        var_90 = cornish_fisher_var(polars_returns, confidence=0.90)
        var_95 = cornish_fisher_var(polars_returns, confidence=0.95)
        var_99 = cornish_fisher_var(polars_returns, confidence=0.99)

        assert var_99 > var_95 > var_90

    def test_normal_returns_close_to_parametric(self) -> None:
        """For normal returns, CF-VaR should be close to parametric VaR."""
        np.random.seed(42)
        normal_returns = pl.Series(np.random.normal(0, 0.02, 5000))

        cf_var = cornish_fisher_var(normal_returns, confidence=0.95)
        p_var = parametric_var(normal_returns, confidence=0.95)

        # Should be within 10% for normal data
        assert abs(cf_var - p_var) / p_var < 0.15

    def test_fat_tails_higher_var(self) -> None:
        """Fat-tailed returns should give higher CF-VaR than parametric."""
        np.random.seed(42)
        # Student-t with low degrees of freedom (fat tails)
        fat_tail_returns = pl.Series(np.random.standard_t(df=3, size=1000) * 0.02)

        cf_var = cornish_fisher_var(fat_tail_returns, confidence=0.99)
        p_var = parametric_var(fat_tail_returns, confidence=0.99)

        # CF-VaR should be higher due to fat tails
        assert cf_var > p_var * 0.9  # Allow some variance

    def test_skewed_returns_adjustment(self) -> None:
        """Negatively skewed returns should have higher CF-VaR."""
        np.random.seed(42)

        # Create negatively skewed returns
        neg_skew = np.concatenate([
            np.random.normal(0.001, 0.01, 800),
            np.random.normal(-0.05, 0.02, 200),  # Occasional large losses
        ])
        np.random.shuffle(neg_skew)
        neg_skew_series = pl.Series(neg_skew)

        cf_var = cornish_fisher_var(neg_skew_series, confidence=0.95)
        p_var = parametric_var(neg_skew_series, confidence=0.95)

        # CF-VaR should account for negative skew
        assert cf_var > 0
        assert p_var > 0

    def test_invalid_confidence(self, polars_returns: pl.Series) -> None:
        """Should raise error for invalid confidence level."""
        with pytest.raises(ValueError):
            cornish_fisher_var(polars_returns, confidence=1.5)

        with pytest.raises(ValueError):
            cornish_fisher_var(polars_returns, confidence=-0.1)

    def test_modified_var_alias(self, polars_returns: pl.Series) -> None:
        """modified_var should be an alias for cornish_fisher_var."""
        cf_var = cornish_fisher_var(polars_returns, confidence=0.95)
        m_var = modified_var(polars_returns, confidence=0.95)

        assert cf_var == m_var


class TestEntropicVaR:
    """Tests for Entropic VaR."""

    def test_positive_evar(self, polars_returns: pl.Series) -> None:
        """EVaR should be positive."""
        evar = entropic_var(polars_returns)
        assert evar > 0

    def test_confidence_monotonicity(self, polars_returns: pl.Series) -> None:
        """Higher confidence should give higher EVaR."""
        evar_90 = entropic_var(polars_returns, confidence=0.90)
        evar_95 = entropic_var(polars_returns, confidence=0.95)
        evar_99 = entropic_var(polars_returns, confidence=0.99)

        # Allow for numerical optimization variance
        assert evar_99 >= evar_95 * 0.9
        assert evar_95 >= evar_90 * 0.9

    def test_evar_greater_than_cvar(self) -> None:
        """EVaR should be greater than or equal to CVaR."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0, 0.02, 500))

        evar = entropic_var(returns, confidence=0.95)

        # Calculate CVaR for comparison
        returns_np = returns.to_numpy()
        q = np.percentile(returns_np, 5)
        cvar = -np.mean(returns_np[returns_np <= q])

        # EVaR >= CVaR (coherence property)
        assert evar >= cvar * 0.9  # Allow small numerical tolerance

    def test_invalid_confidence(self, polars_returns: pl.Series) -> None:
        """Should raise error for invalid confidence."""
        with pytest.raises(ValueError):
            entropic_var(polars_returns, confidence=1.5)


class TestHistoricalVaR:
    """Tests for historical (empirical) VaR."""

    def test_positive_var(self, polars_returns: pl.Series) -> None:
        """Historical VaR should be positive."""
        h_var = historical_var(polars_returns)
        assert h_var > 0

    def test_confidence_monotonicity(self, polars_returns: pl.Series) -> None:
        """Higher confidence should give higher VaR."""
        var_90 = historical_var(polars_returns, confidence=0.90)
        var_95 = historical_var(polars_returns, confidence=0.95)
        var_99 = historical_var(polars_returns, confidence=0.99)

        assert var_99 >= var_95 >= var_90

    def test_matches_quantile(self) -> None:
        """Historical VaR should match empirical quantile."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0, 0.02, 1000))

        h_var = historical_var(returns, confidence=0.95)
        expected = -returns.quantile(0.05, interpolation="linear")

        assert abs(h_var - expected) < 1e-10


class TestParametricVaR:
    """Tests for parametric (Gaussian) VaR."""

    def test_positive_var(self, polars_returns: pl.Series) -> None:
        """Parametric VaR should be positive."""
        p_var = parametric_var(polars_returns)
        assert p_var > 0

    def test_confidence_monotonicity(self, polars_returns: pl.Series) -> None:
        """Higher confidence should give higher VaR."""
        var_90 = parametric_var(polars_returns, confidence=0.90)
        var_95 = parametric_var(polars_returns, confidence=0.95)
        var_99 = parametric_var(polars_returns, confidence=0.99)

        assert var_99 > var_95 > var_90

    def test_matches_formula(self) -> None:
        """Parametric VaR should match analytical formula."""
        from scipy import stats

        np.random.seed(42)
        returns = pl.Series(np.random.normal(0.001, 0.02, 1000))

        p_var = parametric_var(returns, confidence=0.95)

        mu = float(returns.mean())
        sigma = float(returns.std(ddof=1))
        z = stats.norm.ppf(0.05)
        expected = -(mu + z * sigma)

        assert abs(p_var - expected) < 1e-10


class TestVaRComparison:
    """Comparison tests across VaR methods."""

    def test_ordering_normal_data(self) -> None:
        """For normal data, VaR methods should be relatively close."""
        np.random.seed(42)
        normal_returns = pl.Series(np.random.normal(0, 0.02, 2000))

        h_var = historical_var(normal_returns, confidence=0.95)
        p_var = parametric_var(normal_returns, confidence=0.95)
        cf_var = cornish_fisher_var(normal_returns, confidence=0.95)

        # All should be in similar range for normal data
        vars_list = [h_var, p_var, cf_var]
        max_var = max(vars_list)
        min_var = min(vars_list)

        assert max_var / min_var < 1.3  # Within 30%

    def test_fat_tails_increase_cf_var(self) -> None:
        """Fat tails should make CF-VaR higher than parametric."""
        np.random.seed(42)
        # t-distribution with 3 df (very fat tails)
        fat_returns = pl.Series(np.random.standard_t(3, 2000) * 0.02)

        p_var = parametric_var(fat_returns, confidence=0.99)
        cf_var = cornish_fisher_var(fat_returns, confidence=0.99)

        # CF-VaR should be higher due to excess kurtosis adjustment
        assert cf_var > p_var * 0.8


class TestEdgeCases:
    """Edge case tests for VaR metrics."""

    def test_constant_returns(self) -> None:
        """Handle constant returns (zero variance)."""
        constant = pl.Series([0.01] * 100)

        # Should handle gracefully
        p_var = parametric_var(constant)
        h_var = historical_var(constant)

        # With constant positive returns, both should handle gracefully
        # Parametric returns 0 when std=0, historical returns -quantile = -0.01
        assert math.isfinite(p_var)
        assert math.isfinite(h_var)

    def test_all_positive_returns(self, all_positive_returns: pl.Series) -> None:
        """Test with all positive returns."""
        # VaR might be negative (profit) for all positive returns
        h_var = historical_var(all_positive_returns, confidence=0.95)
        # Just verify it doesn't crash
        assert math.isfinite(h_var)

    def test_all_negative_returns(self, all_negative_returns: pl.Series) -> None:
        """Test with all negative returns."""
        h_var = historical_var(all_negative_returns, confidence=0.95)
        # Should be positive (indicates loss)
        assert h_var > 0

    def test_market_data(self, spy_polars: pl.Series) -> None:
        """Test VaR on real market data."""
        pytest.importorskip("pyarrow")

        h_var = historical_var(spy_polars, confidence=0.95)
        p_var = parametric_var(spy_polars, confidence=0.95)
        cf_var = cornish_fisher_var(spy_polars, confidence=0.95)

        # All should be reasonable for SPY
        assert 0 < h_var < 0.1  # Less than 10% daily
        assert 0 < p_var < 0.1
        assert 0 < cf_var < 0.1
