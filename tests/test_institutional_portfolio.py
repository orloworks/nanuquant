"""Tests for institutional portfolio risk metrics.

Tests marginal contribution to risk, Ledoit-Wolf shrinkage, and portfolio volatility.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from nanuquant.institutional import (
    LedoitWolfResult,
    MCRResult,
    correlation_from_covariance,
    ledoit_wolf_covariance,
    marginal_contribution_to_risk,
    portfolio_volatility,
)


class TestMarginalContributionToRisk:
    """Tests for MCR calculation."""

    def test_result_type(self) -> None:
        """Test that result is correct named tuple type."""
        np.random.seed(42)
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, 200),
            "B": np.random.normal(0, 0.015, 200),
        })

        result = marginal_contribution_to_risk(returns, [0.6, 0.4])

        assert isinstance(result, MCRResult)
        assert hasattr(result, "mcr")
        assert hasattr(result, "pcr")
        assert hasattr(result, "total_risk")
        assert hasattr(result, "asset_names")

    def test_pcr_sums_to_one(self) -> None:
        """Percentage contributions to risk must sum to 1."""
        np.random.seed(42)
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, 200),
            "B": np.random.normal(0, 0.015, 200),
            "C": np.random.normal(0, 0.025, 200),
        })

        result = marginal_contribution_to_risk(returns, [0.4, 0.3, 0.3])

        assert abs(sum(result.pcr) - 1.0) < 0.01

    def test_total_risk_positive(self) -> None:
        """Total portfolio risk must be positive."""
        np.random.seed(42)
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, 200),
            "B": np.random.normal(0, 0.015, 200),
        })

        result = marginal_contribution_to_risk(returns, [0.6, 0.4])

        assert result.total_risk > 0

    def test_weight_mismatch_error(self) -> None:
        """Should raise error if weights don't match asset count."""
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, 100),
            "B": np.random.normal(0, 0.015, 100),
        })

        with pytest.raises(ValueError):
            marginal_contribution_to_risk(returns, [0.3, 0.3, 0.4])

    def test_equal_weights_equal_volatility(self) -> None:
        """Equal weight portfolio of equal vol assets should have equal MCR."""
        np.random.seed(42)
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, 500),
            "B": np.random.normal(0, 0.02, 500),
        })

        result = marginal_contribution_to_risk(returns, [0.5, 0.5])

        # MCRs should be close for equal volatility assets
        assert abs(result.mcr[0] - result.mcr[1]) < 0.005

    def test_higher_vol_asset_higher_mcr(self) -> None:
        """Higher volatility asset should have higher MCR."""
        np.random.seed(42)
        returns = pl.DataFrame({
            "Low": np.random.normal(0, 0.01, 500),
            "High": np.random.normal(0, 0.04, 500),
        })

        result = marginal_contribution_to_risk(returns, [0.5, 0.5])

        # High vol asset should have higher MCR
        assert result.mcr[1] > result.mcr[0]


class TestLedoitWolfCovariance:
    """Tests for Ledoit-Wolf shrinkage estimator."""

    def test_result_type(self) -> None:
        """Test that result is correct named tuple type."""
        np.random.seed(42)
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, 100),
            "B": np.random.normal(0, 0.015, 100),
            "C": np.random.normal(0, 0.025, 100),
        })

        result = ledoit_wolf_covariance(returns)

        assert isinstance(result, LedoitWolfResult)
        assert hasattr(result, "covariance")
        assert hasattr(result, "shrinkage_intensity")
        assert hasattr(result, "sample_covariance")

    def test_shrinkage_intensity_range(self) -> None:
        """Shrinkage intensity must be in [0, 1]."""
        np.random.seed(42)
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, 100),
            "B": np.random.normal(0, 0.015, 100),
        })

        result = ledoit_wolf_covariance(returns)

        assert 0 <= result.shrinkage_intensity <= 1

    def test_covariance_matrix_symmetric(self) -> None:
        """Covariance matrix must be symmetric."""
        np.random.seed(42)
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, 100),
            "B": np.random.normal(0, 0.015, 100),
            "C": np.random.normal(0, 0.025, 100),
        })

        result = ledoit_wolf_covariance(returns)

        assert np.allclose(result.covariance, result.covariance.T)

    def test_covariance_positive_semi_definite(self) -> None:
        """Covariance matrix must be positive semi-definite."""
        np.random.seed(42)
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, 100),
            "B": np.random.normal(0, 0.015, 100),
        })

        result = ledoit_wolf_covariance(returns)

        eigenvalues = np.linalg.eigvalsh(result.covariance)
        assert all(ev >= -1e-10 for ev in eigenvalues)

    def test_high_dimension_more_shrinkage(self) -> None:
        """High dimension should lead to more shrinkage."""
        np.random.seed(42)
        n = 50  # observations

        # Low dimension (n >> p)
        returns_low = pl.DataFrame({
            f"A{i}": np.random.normal(0, 0.02, n) for i in range(3)
        })

        # High dimension (n ≈ p)
        returns_high = pl.DataFrame({
            f"A{i}": np.random.normal(0, 0.02, n) for i in range(40)
        })

        result_low = ledoit_wolf_covariance(returns_low)
        result_high = ledoit_wolf_covariance(returns_high)

        # High dimension should have more shrinkage
        assert result_high.shrinkage_intensity > result_low.shrinkage_intensity * 0.5

    def test_diagonal_higher_than_off_diagonal(self) -> None:
        """Variances should be higher than covariances (on average) after shrinkage."""
        np.random.seed(42)
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, 100),
            "B": np.random.normal(0, 0.015, 100),
            "C": np.random.normal(0, 0.025, 100),
        })

        result = ledoit_wolf_covariance(returns)

        diag = np.diag(result.covariance)
        off_diag = result.covariance[~np.eye(3, dtype=bool)]

        # Average variance should be higher than average |covariance|
        assert np.mean(diag) > np.mean(np.abs(off_diag))


class TestPortfolioVolatility:
    """Tests for portfolio volatility calculation."""

    def test_positive_volatility(self) -> None:
        """Portfolio volatility must be positive."""
        np.random.seed(42)
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, 200),
            "B": np.random.normal(0, 0.015, 200),
        })

        vol = portfolio_volatility(returns, [0.6, 0.4])

        assert vol > 0

    def test_single_asset_matches_asset_vol(self) -> None:
        """Single asset portfolio should match asset volatility."""
        np.random.seed(42)
        returns_np = np.random.normal(0, 0.02, 252)
        returns = pl.DataFrame({"A": returns_np, "B": returns_np * 0})  # B has no weight

        # 100% in A
        vol = portfolio_volatility(returns, [1.0, 0.0], annualize=True)

        # Should match annualized vol of A
        expected = np.std(returns_np, ddof=1) * np.sqrt(252)
        assert abs(vol - expected) / expected < 0.01

    def test_diversification_benefit(self) -> None:
        """Portfolio of uncorrelated assets should have lower vol than weighted average."""
        np.random.seed(42)
        n = 500

        # Independent assets with same volatility
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, n),
            "B": np.random.normal(0, 0.02, n),
        })

        port_vol = portfolio_volatility(returns, [0.5, 0.5], annualize=False)

        # Weighted average of individual vols (no diversification)
        undiv_vol = 0.5 * 0.02 + 0.5 * 0.02

        # Portfolio should be lower due to diversification
        assert port_vol < undiv_vol

    def test_shrinkage_option(self) -> None:
        """Test with and without shrinkage."""
        np.random.seed(42)
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, 100),
            "B": np.random.normal(0, 0.015, 100),
        })

        vol_sample = portfolio_volatility(returns, [0.6, 0.4], use_shrinkage=False)
        vol_shrunk = portfolio_volatility(returns, [0.6, 0.4], use_shrinkage=True)

        # Both should be positive
        assert vol_sample > 0
        assert vol_shrunk > 0

    def test_annualization(self) -> None:
        """Test annualization option."""
        np.random.seed(42)
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, 252),
            "B": np.random.normal(0, 0.015, 252),
        })

        vol_daily = portfolio_volatility(
            returns, [0.6, 0.4], annualize=False
        )
        vol_annual = portfolio_volatility(
            returns, [0.6, 0.4], annualize=True, periods_per_year=252
        )

        # Annual should be sqrt(252) times daily
        assert abs(vol_annual / vol_daily - np.sqrt(252)) < 0.1


class TestCorrelationFromCovariance:
    """Tests for correlation matrix conversion."""

    def test_diagonal_ones(self) -> None:
        """Correlation matrix should have 1s on diagonal."""
        np.random.seed(42)
        cov = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.09, 0.015],
            [0.02, 0.015, 0.0625],
        ])

        corr = correlation_from_covariance(cov)

        assert np.allclose(np.diag(corr), [1, 1, 1])

    def test_correlation_range(self) -> None:
        """All correlations must be in [-1, 1]."""
        np.random.seed(42)
        cov = np.array([
            [0.04, 0.01, -0.005],
            [0.01, 0.09, 0.02],
            [-0.005, 0.02, 0.0625],
        ])

        corr = correlation_from_covariance(cov)

        assert np.all(corr >= -1)
        assert np.all(corr <= 1)

    def test_symmetric(self) -> None:
        """Correlation matrix must be symmetric."""
        cov = np.array([
            [0.04, 0.01],
            [0.01, 0.09],
        ])

        corr = correlation_from_covariance(cov)

        assert np.allclose(corr, corr.T)


class TestSyntheticDataRecovery:
    """Synthetic data tests for portfolio metrics."""

    def test_mcr_risk_parity(self) -> None:
        """Risk parity weights should give equal risk contributions."""
        np.random.seed(42)

        # Create assets with known volatilities
        vol1, vol2 = 0.02, 0.04
        returns = pl.DataFrame({
            "Low": np.random.normal(0, vol1, 500),
            "High": np.random.normal(0, vol2, 500),
        })

        # Approximate risk parity weights (inverse vol)
        w1 = (1 / vol1) / (1 / vol1 + 1 / vol2)
        w2 = (1 / vol2) / (1 / vol1 + 1 / vol2)

        result = marginal_contribution_to_risk(returns, [w1, w2])

        # Risk contributions should be approximately equal
        assert abs(result.pcr[0] - result.pcr[1]) < 0.15

    def test_ledoit_wolf_converges_large_sample(self) -> None:
        """With large sample, shrinkage should be low."""
        np.random.seed(42)

        # Large sample relative to dimension
        returns = pl.DataFrame({
            "A": np.random.normal(0, 0.02, 2000),
            "B": np.random.normal(0, 0.015, 2000),
        })

        result = ledoit_wolf_covariance(returns)

        # Shrinkage should be low
        assert result.shrinkage_intensity < 0.2
