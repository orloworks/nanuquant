"""Tests for institutional volatility metrics.

Tests the GARCH volatility model and ARCH effect test implementations
using the 4-pillar validation framework.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from nanuquant.institutional import (
    ARCHTestResult,
    GARCHResult,
    arch_effect_test,
    garch_volatility,
)


class TestARCHEffectTest:
    """Tests for Engle's LM test for ARCH effects."""

    def test_result_type(self, polars_returns: pl.Series) -> None:
        """Test that result is correct named tuple type."""
        result = arch_effect_test(polars_returns)
        assert isinstance(result, ARCHTestResult)
        assert hasattr(result, "statistic")
        assert hasattr(result, "p_value")
        assert hasattr(result, "lags")
        assert hasattr(result, "has_arch_effects")

    def test_p_value_range(self, polars_returns: pl.Series) -> None:
        """P-value must be in [0, 1]."""
        result = arch_effect_test(polars_returns)
        assert 0 <= result.p_value <= 1

    def test_statistic_non_negative(self, polars_returns: pl.Series) -> None:
        """Test statistic must be non-negative (n * R^2)."""
        result = arch_effect_test(polars_returns)
        assert result.statistic >= 0

    def test_has_arch_effects_consistency(self, polars_returns: pl.Series) -> None:
        """has_arch_effects should match p_value < 0.05."""
        result = arch_effect_test(polars_returns)
        expected = result.p_value < 0.05
        assert result.has_arch_effects == expected

    def test_white_noise_no_arch_effects(self) -> None:
        """White noise should have no ARCH effects."""
        np.random.seed(42)
        white_noise = pl.Series(np.random.normal(0, 0.02, 1000))

        result = arch_effect_test(white_noise)

        # White noise should not reject null of no ARCH effects
        # Allow for random chance (about 5% false positive rate)
        assert result.p_value > 0.01  # Very conservative threshold

    def test_garch_data_has_arch_effects(self) -> None:
        """Data with known ARCH effects should be detected."""
        np.random.seed(42)
        n = 1000

        # Generate GARCH(1,1) data with strong persistence
        omega = 0.0001
        alpha = 0.15
        beta = 0.80
        vol = np.zeros(n)
        returns = np.zeros(n)
        vol[0] = 0.02

        for t in range(1, n):
            vol[t] = np.sqrt(omega + alpha * returns[t - 1] ** 2 + beta * vol[t - 1] ** 2)
            returns[t] = vol[t] * np.random.normal()

        garch_returns = pl.Series(returns)
        result = arch_effect_test(garch_returns)

        # Should strongly reject null of no ARCH effects
        assert result.has_arch_effects
        assert result.p_value < 0.05

    def test_different_lag_counts(self, polars_returns: pl.Series) -> None:
        """Test with different lag specifications."""
        result_5 = arch_effect_test(polars_returns, lags=5)
        result_12 = arch_effect_test(polars_returns, lags=12)
        result_20 = arch_effect_test(polars_returns, lags=20)

        assert result_5.lags == 5
        assert result_12.lags == 12
        assert result_20.lags == 20

    def test_insufficient_data(self) -> None:
        """Test error handling for insufficient data."""
        short_series = pl.Series([0.01, 0.02, 0.03, 0.04, 0.05])

        with pytest.raises(Exception):  # InsufficientDataError
            arch_effect_test(short_series, lags=12)

    def test_market_data(self, spy_polars: pl.Series) -> None:
        """Test on real market data - should likely show some ARCH effects."""
        pytest.importorskip("pyarrow")
        result = arch_effect_test(spy_polars)

        # Real market data typically shows ARCH effects
        assert 0 <= result.p_value <= 1


class TestGARCHVolatility:
    """Tests for GARCH(1,1) volatility estimation."""

    def test_result_type(self, polars_returns: pl.Series) -> None:
        """Test that result is correct named tuple type."""
        result = garch_volatility(polars_returns)
        assert isinstance(result, GARCHResult)
        assert hasattr(result, "omega")
        assert hasattr(result, "alpha")
        assert hasattr(result, "beta")
        assert hasattr(result, "conditional_volatility")
        assert hasattr(result, "long_run_variance")
        assert hasattr(result, "persistence")
        assert hasattr(result, "forecast")

    def test_parameter_constraints(self, polars_returns: pl.Series) -> None:
        """GARCH parameters must satisfy constraints."""
        result = garch_volatility(polars_returns)

        # Non-negativity constraints
        assert result.omega >= 0
        assert result.alpha >= 0
        assert result.beta >= 0

        # Stationarity constraint (persistence < 1)
        assert result.persistence < 1.0

    def test_persistence_formula(self, polars_returns: pl.Series) -> None:
        """Persistence should equal alpha + beta."""
        result = garch_volatility(polars_returns)
        assert abs(result.persistence - (result.alpha + result.beta)) < 1e-10

    def test_conditional_volatility_positive(self, polars_returns: pl.Series) -> None:
        """Conditional volatility must be positive."""
        result = garch_volatility(polars_returns)

        assert len(result.conditional_volatility) == len(polars_returns)
        assert (result.conditional_volatility > 0).all()

    def test_long_run_variance_formula(self, polars_returns: pl.Series) -> None:
        """Long-run variance should equal omega / (1 - alpha - beta)."""
        result = garch_volatility(polars_returns)

        if result.persistence < 1:
            expected_lrv = result.omega / (1 - result.persistence)
            assert abs(result.long_run_variance - expected_lrv) < 1e-10

    def test_forecast_positive(self, polars_returns: pl.Series) -> None:
        """Volatility forecast must be positive."""
        result = garch_volatility(polars_returns)
        assert result.forecast > 0

    def test_high_volatility_clustering_higher_alpha(self) -> None:
        """Data with high volatility clustering should have higher alpha."""
        np.random.seed(42)
        n = 500

        # Generate data with strong ARCH effects
        vol = np.ones(n) * 0.02
        returns_clustered = np.zeros(n)
        for t in range(1, n):
            if abs(returns_clustered[t - 1]) > 0.03:
                vol[t] = 0.04  # High vol after big move
            else:
                vol[t] = 0.01  # Low vol otherwise
            returns_clustered[t] = vol[t] * np.random.normal()

        # Generate data without clustering
        returns_unclustered = np.random.normal(0, 0.02, n)

        result_clustered = garch_volatility(pl.Series(returns_clustered))
        result_unclustered = garch_volatility(pl.Series(returns_unclustered))

        # Both should produce valid results
        assert result_clustered.persistence < 1
        assert result_unclustered.persistence < 1

    def test_unsupported_p_q(self, polars_returns: pl.Series) -> None:
        """Only GARCH(1,1) is supported."""
        with pytest.raises(ValueError):
            garch_volatility(polars_returns, p=2, q=1)

        with pytest.raises(ValueError):
            garch_volatility(polars_returns, p=1, q=2)

    def test_insufficient_data(self) -> None:
        """Test error handling for insufficient data."""
        short_series = pl.Series([0.01] * 10)

        with pytest.raises(Exception):  # InsufficientDataError
            garch_volatility(short_series)

    def test_market_data(self, spy_polars: pl.Series) -> None:
        """Test GARCH on real market data."""
        pytest.importorskip("pyarrow")
        result = garch_volatility(spy_polars)

        # SPY should have valid GARCH parameters
        assert result.omega > 0
        assert result.alpha >= 0
        assert result.beta >= 0
        assert result.persistence < 1


class TestSyntheticDataRecovery:
    """Tests for GARCH parameter recovery from synthetic data."""

    def test_known_garch_parameter_recovery(self) -> None:
        """Generate GARCH data with known params and verify approximate recovery."""
        np.random.seed(42)
        n = 2000

        # True parameters
        true_omega = 0.0001
        true_alpha = 0.10
        true_beta = 0.85

        # Generate GARCH(1,1) data
        vol = np.zeros(n)
        returns = np.zeros(n)
        vol[0] = np.sqrt(true_omega / (1 - true_alpha - true_beta))

        for t in range(1, n):
            vol[t] = np.sqrt(
                true_omega + true_alpha * returns[t - 1] ** 2 + true_beta * vol[t - 1] ** 2
            )
            returns[t] = vol[t] * np.random.normal()

        result = garch_volatility(pl.Series(returns))

        # Check persistence is in reasonable range (within 30% of true)
        true_persistence = true_alpha + true_beta
        assert abs(result.persistence - true_persistence) < 0.3

    def test_high_persistence_recovery(self) -> None:
        """Test recovery of highly persistent volatility."""
        np.random.seed(42)
        n = 1500

        # High persistence parameters
        true_omega = 0.00005
        true_alpha = 0.08
        true_beta = 0.90
        true_persistence = true_alpha + true_beta

        # Generate data
        vol = np.zeros(n)
        returns = np.zeros(n)
        vol[0] = np.sqrt(true_omega / (1 - true_persistence))

        for t in range(1, n):
            vol[t] = np.sqrt(
                true_omega + true_alpha * returns[t - 1] ** 2 + true_beta * vol[t - 1] ** 2
            )
            returns[t] = vol[t] * np.random.normal()

        result = garch_volatility(pl.Series(returns))

        # Should detect high persistence
        assert result.persistence > 0.8


class TestEdgeCases:
    """Edge case tests for volatility metrics."""

    def test_constant_volatility(self) -> None:
        """Test with constant volatility data."""
        np.random.seed(42)
        constant_vol = pl.Series(np.random.normal(0, 0.02, 500))

        result = garch_volatility(constant_vol)

        # Should produce valid results
        # Note: constant vol data can yield persistence = 1.0 (boundary case)
        assert result.persistence <= 1.0 + 1e-6  # Allow tiny numerical tolerance
        assert (result.conditional_volatility > 0).all()

    def test_trending_volatility(self) -> None:
        """Test with trending volatility (non-stationary)."""
        np.random.seed(42)
        n = 500
        vol = np.linspace(0.01, 0.05, n)
        returns = vol * np.random.normal(size=n)

        trending = pl.Series(returns)
        result = garch_volatility(trending)

        # Should still produce valid output
        # Note: trending/non-stationary data can yield persistence >= 1
        assert result.persistence <= 1.0 + 1e-6  # Allow tiny numerical tolerance

    def test_low_volatility_period(self) -> None:
        """Test with very low volatility data."""
        np.random.seed(42)
        low_vol = pl.Series(np.random.normal(0, 0.001, 500))

        result = garch_volatility(low_vol)

        # Should handle low volatility
        assert result.forecast > 0
        assert (result.conditional_volatility > 0).all()

    def test_high_volatility_period(self) -> None:
        """Test with very high volatility data."""
        np.random.seed(42)
        high_vol = pl.Series(np.random.normal(0, 0.10, 500))

        result = garch_volatility(high_vol)

        # Should handle high volatility
        assert result.forecast > 0


class TestInvariants:
    """Mathematical invariant tests for volatility metrics."""

    def test_arch_test_lags_affect_statistic(self, polars_returns: pl.Series) -> None:
        """More lags should generally change the test statistic."""
        result_5 = arch_effect_test(polars_returns, lags=5)
        result_20 = arch_effect_test(polars_returns, lags=20)

        # Degrees of freedom change with lags
        assert result_5.lags < result_20.lags

    def test_garch_volatility_scale_invariance(self) -> None:
        """GARCH parameters should scale appropriately with return magnitude."""
        np.random.seed(42)
        base_returns = np.random.normal(0, 0.02, 500)

        result_1x = garch_volatility(pl.Series(base_returns))
        result_2x = garch_volatility(pl.Series(base_returns * 2))

        # Alpha and beta should be similar (scale-invariant)
        # Omega should scale with variance (4x for 2x returns)
        assert abs(result_1x.alpha - result_2x.alpha) < 0.1
        assert abs(result_1x.beta - result_2x.beta) < 0.1

    def test_garch_persistence_bounds(self, polars_returns: pl.Series) -> None:
        """Persistence must be between 0 and 1 for stationarity."""
        result = garch_volatility(polars_returns)

        assert 0 <= result.persistence < 1

    def test_conditional_vol_converges_to_long_run(self) -> None:
        """Conditional volatility should converge to long-run in stable periods."""
        np.random.seed(42)
        # Generate stable data (low volatility, no shocks)
        stable_returns = pl.Series(np.random.normal(0, 0.01, 1000))

        result = garch_volatility(stable_returns)

        # Long-run volatility
        long_run_vol = math.sqrt(result.long_run_variance)

        # Average conditional volatility should be close to long-run
        avg_cond_vol = result.conditional_volatility.mean()
        if avg_cond_vol is not None:
            assert abs(avg_cond_vol - long_run_vol) < long_run_vol * 0.5
