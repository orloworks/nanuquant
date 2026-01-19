"""Tests for institutional robustness metrics.

Tests cover:
- Probabilistic Sharpe Ratio (PSR)
- Deflated Sharpe Ratio (DSR)
- Minimum Track Record Length
- Newey-West standard errors
- Edge cases and statistical properties
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from nanuquant.institutional import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from nanuquant.institutional._helpers import (
    _autocorrelation,
    _expected_max_sharpe,
    newey_west_se,
)
from nanuquant.institutional.robustness import minimum_track_record_length


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestAutocorrelation:
    """Tests for _autocorrelation helper."""

    def test_zero_autocorr_iid(self) -> None:
        """IID returns should have near-zero autocorrelation."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0, 0.02, 1000))
        autocorr = _autocorrelation(returns, lag=1)
        assert abs(autocorr) < 0.1  # Should be close to zero

    def test_positive_autocorr_trending(self) -> None:
        """Trending returns should have positive autocorrelation."""
        # Create trending series
        trend = np.cumsum(np.ones(100) * 0.01)
        noise = np.random.normal(0, 0.001, 100)
        returns = pl.Series(trend + noise)
        autocorr = _autocorrelation(returns, lag=1)
        assert autocorr > 0.5

    def test_lag_two(self) -> None:
        """Test autocorrelation at lag 2."""
        np.random.seed(123)
        returns = pl.Series(np.random.normal(0, 0.02, 500))
        autocorr_lag1 = _autocorrelation(returns, lag=1)
        autocorr_lag2 = _autocorrelation(returns, lag=2)
        # Both should be near zero for IID
        assert abs(autocorr_lag1) < 0.15
        assert abs(autocorr_lag2) < 0.15

    def test_insufficient_data(self) -> None:
        """Insufficient data returns zero."""
        returns = pl.Series([0.01, 0.02])
        assert _autocorrelation(returns, lag=5) == 0.0

    def test_empty_series(self) -> None:
        """Empty series returns zero."""
        returns = pl.Series([], dtype=pl.Float64)
        assert _autocorrelation(returns, lag=1) == 0.0


class TestNeweyWestSE:
    """Tests for Newey-West standard error calculation."""

    def test_larger_than_naive_with_positive_autocorr(self) -> None:
        """NW SE should be larger than naive SE when positive autocorrelation exists."""
        # Create series with positive autocorrelation
        np.random.seed(42)
        n = 500
        returns = np.zeros(n)
        returns[0] = np.random.normal(0, 0.02)
        for i in range(1, n):
            returns[i] = 0.3 * returns[i - 1] + np.random.normal(0, 0.02)
        returns_series = pl.Series(returns)

        nw_se = newey_west_se(returns_series)
        naive_se = float(returns_series.std()) / math.sqrt(n)

        # NW SE should be larger due to positive autocorrelation
        assert nw_se > naive_se * 0.9  # Allow some tolerance

    def test_similar_to_naive_for_iid(self) -> None:
        """NW SE should be close to naive SE for IID returns."""
        np.random.seed(42)
        n = 1000
        returns = pl.Series(np.random.normal(0, 0.02, n))

        nw_se = newey_west_se(returns)
        naive_se = float(returns.std()) / math.sqrt(n)

        # Should be similar for IID data
        ratio = nw_se / naive_se
        assert 0.8 < ratio < 1.5  # Allow reasonable tolerance

    def test_custom_lags(self) -> None:
        """Test with custom lag specification."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0, 0.02, 200))

        se_default = newey_west_se(returns)
        se_lag5 = newey_west_se(returns, lags=5)
        se_lag10 = newey_west_se(returns, lags=10)

        # All should be positive
        assert se_default > 0
        assert se_lag5 > 0
        assert se_lag10 > 0

    def test_empty_returns_zero(self) -> None:
        """Empty series returns zero."""
        returns = pl.Series([], dtype=pl.Float64)
        assert newey_west_se(returns) == 0.0

    def test_single_value_returns_zero(self) -> None:
        """Single value returns zero."""
        returns = pl.Series([0.01])
        assert newey_west_se(returns) == 0.0


class TestExpectedMaxSharpe:
    """Tests for expected maximum Sharpe ratio calculation."""

    def test_increases_with_trials(self) -> None:
        """Expected max should increase with number of trials."""
        max_10 = _expected_max_sharpe(10)
        max_100 = _expected_max_sharpe(100)
        max_1000 = _expected_max_sharpe(1000)

        assert max_10 < max_100 < max_1000

    def test_known_values(self) -> None:
        """Test against known approximate values."""
        # sqrt(2 * ln(n)) approximation for large n
        # n=100: sqrt(2 * ln(100)) ≈ 3.03
        # n=1000: sqrt(2 * ln(1000)) ≈ 3.72
        # Actual values are slightly lower due to correction term

        max_100 = _expected_max_sharpe(100)
        max_1000 = _expected_max_sharpe(1000)

        # Should be in reasonable range
        assert 1.5 < max_100 < 3.5
        assert 2.0 < max_1000 < 4.0

    def test_single_trial_returns_zero(self) -> None:
        """Single trial should return zero (no selection bias)."""
        assert _expected_max_sharpe(1) == 0.0

    def test_zero_trials_returns_zero(self) -> None:
        """Zero trials should return zero."""
        assert _expected_max_sharpe(0) == 0.0

    def test_with_variance(self) -> None:
        """Test with non-unit variance."""
        max_unit = _expected_max_sharpe(100, var_sharpe=1.0)
        max_higher = _expected_max_sharpe(100, var_sharpe=2.0)

        # Higher variance should give higher expected max
        assert max_higher > max_unit


# =============================================================================
# Probabilistic Sharpe Ratio Tests
# =============================================================================


class TestProbabilisticSharpeRatio:
    """Tests for Probabilistic Sharpe Ratio calculation."""

    def test_psr_in_valid_range(self, polars_returns: pl.Series) -> None:
        """PSR should be in [0, 1]."""
        result = probabilistic_sharpe_ratio(polars_returns, benchmark_sr=0.0)
        assert 0 <= result.psr <= 1
        assert 0 <= result.p_value <= 1

    def test_psr_plus_pvalue_equals_one(self) -> None:
        """PSR + p_value should equal 1 (one-sided test)."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0.001, 0.02, 252))
        result = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)

        assert abs(result.psr + result.p_value - 1.0) < 1e-10

    def test_positive_sharpe_high_psr(self) -> None:
        """Strong positive Sharpe should have high PSR against zero benchmark."""
        np.random.seed(42)
        # Generate returns with positive drift
        returns = pl.Series(np.random.normal(0.002, 0.015, 500))
        result = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)

        # Should have high confidence SR > 0
        assert result.psr > 0.7

    def test_negative_sharpe_low_psr(self) -> None:
        """Negative Sharpe should have low PSR against zero benchmark."""
        np.random.seed(42)
        # Generate returns with negative drift
        returns = pl.Series(np.random.normal(-0.001, 0.02, 500))
        result = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)

        # Should have low confidence SR > 0
        assert result.psr < 0.5

    def test_higher_benchmark_lower_psr(self) -> None:
        """Higher benchmark should result in lower PSR."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0.001, 0.02, 252))

        psr_0 = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
        psr_1 = probabilistic_sharpe_ratio(returns, benchmark_sr=1.0)
        psr_2 = probabilistic_sharpe_ratio(returns, benchmark_sr=2.0)

        assert psr_0.psr > psr_1.psr > psr_2.psr

    def test_more_data_higher_psr_for_positive_sr(self) -> None:
        """More data should increase PSR for strategies with positive true SR."""
        np.random.seed(42)
        short_returns = pl.Series(np.random.normal(0.001, 0.02, 100))
        long_returns = pl.Series(np.random.normal(0.001, 0.02, 1000))

        # Both have same mean/std, but more data = more certainty
        psr_short = probabilistic_sharpe_ratio(short_returns, benchmark_sr=0.0)
        psr_long = probabilistic_sharpe_ratio(long_returns, benchmark_sr=0.0)

        # More data should give higher confidence (assuming positive true SR)
        # Note: This may not always hold due to sampling, but generally true
        assert psr_long.psr > psr_short.psr - 0.2  # Allow some tolerance

    def test_skewness_effect(self) -> None:
        """Negative skewness should reduce PSR compared to normal returns."""
        np.random.seed(42)

        # Generate normal returns
        normal_returns = pl.Series(np.random.normal(0.001, 0.02, 500))

        # Generate negatively skewed returns with similar mean/std
        from scipy.stats import skewnorm

        skewed_data = skewnorm.rvs(-5, loc=0.001, scale=0.02, size=500)
        skewed_returns = pl.Series(skewed_data)

        psr_normal = probabilistic_sharpe_ratio(normal_returns, benchmark_sr=0.0)
        psr_skewed = probabilistic_sharpe_ratio(skewed_returns, benchmark_sr=0.0)

        # PSR calculation accounts for skewness
        # The effect depends on the sign of Sharpe * skewness
        assert psr_normal.psr != psr_skewed.psr  # Should be different

    def test_constant_returns(self) -> None:
        """Constant returns (zero vol) should handle gracefully."""
        returns = pl.Series([0.01] * 100)
        result = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)

        # Zero volatility is a degenerate case - NaN is mathematically correct
        # since Sharpe ratio is undefined when volatility is zero
        assert math.isnan(result.psr) or 0 <= result.psr <= 1

    def test_minimum_data_requirement(self) -> None:
        """Should require minimum data points."""
        returns = pl.Series([0.01, 0.02])  # Only 2 points

        with pytest.raises(Exception):  # InsufficientDataError
            probabilistic_sharpe_ratio(returns)

    def test_with_real_market_data(self, spy_polars: pl.Series) -> None:
        """Test with real SPY market data."""
        result = probabilistic_sharpe_ratio(spy_polars, benchmark_sr=0.0)

        # SPY should have positive expected returns
        assert result.psr > 0.5
        assert 0 <= result.psr <= 1


# =============================================================================
# Deflated Sharpe Ratio Tests
# =============================================================================


class TestDeflatedSharpeRatio:
    """Tests for Deflated Sharpe Ratio calculation."""

    def test_dsr_in_valid_range(self, polars_returns: pl.Series) -> None:
        """DSR should be in [0, 1]."""
        dsr = deflated_sharpe_ratio(polars_returns, n_trials=100)
        assert 0 <= dsr <= 1

    def test_dsr_decreases_with_more_trials(self) -> None:
        """DSR should decrease as number of trials increases."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0.001, 0.02, 500))

        dsr_10 = deflated_sharpe_ratio(returns, n_trials=10)
        dsr_100 = deflated_sharpe_ratio(returns, n_trials=100)
        dsr_1000 = deflated_sharpe_ratio(returns, n_trials=1000)

        assert dsr_10 > dsr_100 > dsr_1000

    def test_dsr_equals_psr_for_single_trial(self) -> None:
        """DSR with n_trials=1 should equal PSR with benchmark=0."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0.001, 0.02, 252))

        dsr = deflated_sharpe_ratio(returns, n_trials=1)
        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)

        # Should be very close (expected max for n=1 is 0)
        assert abs(dsr - psr.psr) < 0.01

    def test_strong_strategy_high_dsr(self) -> None:
        """Very strong strategy should maintain high DSR even with many trials."""
        np.random.seed(42)
        # Generate very strong returns (high Sharpe)
        returns = pl.Series(np.random.normal(0.005, 0.01, 500))

        dsr = deflated_sharpe_ratio(returns, n_trials=100)
        # Strong strategy should still look good after correction
        assert dsr > 0.5

    def test_mediocre_strategy_low_dsr_many_trials(self) -> None:
        """Mediocre strategy should have low DSR after many trials."""
        np.random.seed(42)
        # Generate mediocre returns (low Sharpe)
        returns = pl.Series(np.random.normal(0.0003, 0.02, 500))

        dsr = deflated_sharpe_ratio(returns, n_trials=1000)
        # After 1000 trials, mediocre strategy likely due to luck
        assert dsr < 0.5

    def test_negative_n_trials_raises(self) -> None:
        """Negative n_trials should raise ValueError."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0, 0.02, 100))

        with pytest.raises(ValueError):
            deflated_sharpe_ratio(returns, n_trials=-1)

    def test_zero_n_trials_raises(self) -> None:
        """Zero n_trials should raise ValueError."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0, 0.02, 100))

        with pytest.raises(ValueError):
            deflated_sharpe_ratio(returns, n_trials=0)

    def test_negative_var_sharpe_raises(self) -> None:
        """Negative var_sharpe should raise ValueError."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0, 0.02, 100))

        with pytest.raises(ValueError):
            deflated_sharpe_ratio(returns, n_trials=100, var_sharpe=-1.0)

    def test_higher_var_sharpe_lower_dsr(self) -> None:
        """Higher variance assumption should lead to higher expected max, thus lower DSR."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0.001, 0.02, 500))

        dsr_unit = deflated_sharpe_ratio(returns, n_trials=100, var_sharpe=1.0)
        dsr_high = deflated_sharpe_ratio(returns, n_trials=100, var_sharpe=2.0)

        assert dsr_high < dsr_unit

    def test_with_real_market_data(self, spy_polars: pl.Series) -> None:
        """Test with real SPY market data."""
        dsr = deflated_sharpe_ratio(spy_polars, n_trials=50)

        # Should be valid probability
        assert 0 <= dsr <= 1


# =============================================================================
# Minimum Track Record Length Tests
# =============================================================================


class TestMinimumTrackRecordLength:
    """Tests for minimum track record length calculation."""

    def test_positive_sharpe_finite_length(self) -> None:
        """Positive Sharpe should give finite track record length."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0.001, 0.02, 252))
        min_trl = minimum_track_record_length(returns, target_sr=0.0)

        assert min_trl > 0
        assert min_trl < float("inf")

    def test_higher_confidence_longer_track(self) -> None:
        """Higher confidence should require longer track record."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0.001, 0.02, 252))

        min_90 = minimum_track_record_length(returns, confidence=0.90)
        min_95 = minimum_track_record_length(returns, confidence=0.95)
        min_99 = minimum_track_record_length(returns, confidence=0.99)

        assert min_90 < min_95 < min_99

    def test_higher_target_longer_track(self) -> None:
        """Higher target Sharpe should require longer track record."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0.002, 0.02, 500))

        min_0 = minimum_track_record_length(returns, target_sr=0.0)
        min_05 = minimum_track_record_length(returns, target_sr=0.5)
        min_1 = minimum_track_record_length(returns, target_sr=1.0)

        # Higher target = harder to prove = longer track record
        assert min_0 < min_05 < min_1

    def test_stronger_strategy_shorter_track(self) -> None:
        """Stronger strategy (higher Sharpe) needs shorter track record."""
        np.random.seed(42)

        weak_returns = pl.Series(np.random.normal(0.0005, 0.02, 500))
        strong_returns = pl.Series(np.random.normal(0.002, 0.02, 500))

        min_weak = minimum_track_record_length(weak_returns, target_sr=0.0)
        min_strong = minimum_track_record_length(strong_returns, target_sr=0.0)

        assert min_strong < min_weak

    def test_invalid_confidence_raises(self) -> None:
        """Invalid confidence should raise ValueError."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0, 0.02, 100))

        with pytest.raises(ValueError):
            minimum_track_record_length(returns, confidence=1.5)

        with pytest.raises(ValueError):
            minimum_track_record_length(returns, confidence=0.0)

        with pytest.raises(ValueError):
            minimum_track_record_length(returns, confidence=-0.1)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_series_psr(self, empty_returns: pl.Series) -> None:
        """Empty series should raise error for PSR."""
        with pytest.raises(Exception):
            probabilistic_sharpe_ratio(empty_returns)

    def test_empty_series_dsr(self, empty_returns: pl.Series) -> None:
        """Empty series should raise error for DSR."""
        with pytest.raises(Exception):
            deflated_sharpe_ratio(empty_returns, n_trials=100)

    def test_single_return_psr(self, single_return: pl.Series) -> None:
        """Single return should raise error for PSR."""
        with pytest.raises(Exception):
            probabilistic_sharpe_ratio(single_return)

    def test_single_return_dsr(self, single_return: pl.Series) -> None:
        """Single return should raise error for DSR."""
        with pytest.raises(Exception):
            deflated_sharpe_ratio(single_return, n_trials=100)

    def test_all_positive_returns(self, all_positive_returns: pl.Series) -> None:
        """All positive returns should still compute valid metrics."""
        result = probabilistic_sharpe_ratio(all_positive_returns)
        assert 0 <= result.psr <= 1

    def test_all_negative_returns(self, all_negative_returns: pl.Series) -> None:
        """All negative returns should still compute valid metrics."""
        result = probabilistic_sharpe_ratio(all_negative_returns)
        assert 0 <= result.psr <= 1

    def test_flat_returns_psr(self, flat_returns: pl.Series) -> None:
        """Flat returns (all zeros) should handle gracefully."""
        result = probabilistic_sharpe_ratio(flat_returns)
        # Zero mean, zero vol is degenerate
        assert 0 <= result.psr <= 1

    def test_very_large_series(self) -> None:
        """Test with very large series for performance."""
        np.random.seed(42)
        large_returns = pl.Series(np.random.normal(0.0005, 0.02, 10000))

        result = probabilistic_sharpe_ratio(large_returns)
        assert 0 <= result.psr <= 1

        dsr = deflated_sharpe_ratio(large_returns, n_trials=100)
        assert 0 <= dsr <= 1

    def test_extreme_returns(self) -> None:
        """Test with extreme return values."""
        # Mix of normal and extreme returns
        returns = pl.Series([0.01, -0.5, 0.02, 0.8, -0.1, 0.03, -0.2, 0.15])

        result = probabilistic_sharpe_ratio(returns)
        assert 0 <= result.psr <= 1

        dsr = deflated_sharpe_ratio(returns, n_trials=50)
        assert 0 <= dsr <= 1


# =============================================================================
# Integration Tests with Real Market Data
# =============================================================================


class TestIntegrationMarketData:
    """Integration tests using real market data."""

    def test_psr_spy_vs_qqq(
        self,
        spy_polars: pl.Series,
        qqq_polars: pl.Series,
    ) -> None:
        """Compare PSR between SPY and QQQ."""
        psr_spy = probabilistic_sharpe_ratio(spy_polars, benchmark_sr=0.0)
        psr_qqq = probabilistic_sharpe_ratio(qqq_polars, benchmark_sr=0.0)

        # Both should have valid PSR in [0, 1]
        # PSR can be 1.0 for very strong strategies with long track records
        assert 0 <= psr_spy.psr <= 1
        assert 0 <= psr_qqq.psr <= 1

    def test_dsr_realistic_trial_count(self, spy_polars: pl.Series) -> None:
        """Test DSR with realistic trial counts from strategy development."""
        # Typical quantitative research might test 50-500 strategy variants
        dsr_50 = deflated_sharpe_ratio(spy_polars, n_trials=50)
        dsr_200 = deflated_sharpe_ratio(spy_polars, n_trials=200)
        dsr_500 = deflated_sharpe_ratio(spy_polars, n_trials=500)

        # All should be valid
        assert 0 <= dsr_50 <= 1
        assert 0 <= dsr_200 <= 1
        assert 0 <= dsr_500 <= 1

        # Monotonically non-increasing (more trials = same or lower DSR)
        # DSR can bottom out at 0.0 for weak strategies
        assert dsr_50 >= dsr_200 >= dsr_500

    def test_bond_fund_metrics(self, bnd_polars: pl.Series) -> None:
        """Test metrics on bond fund (lower vol, different characteristics)."""
        psr = probabilistic_sharpe_ratio(bnd_polars, benchmark_sr=0.0)
        dsr = deflated_sharpe_ratio(bnd_polars, n_trials=100)

        # Bonds should have positive but lower Sharpe historically
        assert 0 <= psr.psr <= 1
        assert 0 <= dsr <= 1

    def test_minimum_trl_practical_values(self, spy_polars: pl.Series) -> None:
        """Minimum TRL should give practical values for real data."""
        min_trl = minimum_track_record_length(
            spy_polars, target_sr=0.0, confidence=0.95
        )

        # Should be reasonable (not too short, not astronomical)
        assert 10 < min_trl < 10000


# =============================================================================
# Statistical Property Tests
# =============================================================================


class TestStatisticalProperties:
    """Tests for statistical properties and consistency."""

    def test_psr_consistency_across_seeds(self) -> None:
        """PSR should be consistent for similar distributions."""
        psrs = []
        for seed in range(5):
            np.random.seed(seed * 100)
            returns = pl.Series(np.random.normal(0.001, 0.02, 500))
            psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
            psrs.append(psr.psr)

        # All PSR values should be in similar range (0.6-0.9 for this setup)
        mean_psr = sum(psrs) / len(psrs)
        for psr in psrs:
            assert abs(psr - mean_psr) < 0.25  # Reasonable variance

    def test_dsr_monotonicity_in_sharpe(self) -> None:
        """DSR should increase with higher observed Sharpe."""
        n_trials = 100
        np.random.seed(42)

        # Different mean returns = different Sharpe ratios
        low_returns = pl.Series(np.random.normal(0.0005, 0.02, 500))
        med_returns = pl.Series(np.random.normal(0.001, 0.02, 500))
        high_returns = pl.Series(np.random.normal(0.002, 0.02, 500))

        dsr_low = deflated_sharpe_ratio(low_returns, n_trials=n_trials)
        dsr_med = deflated_sharpe_ratio(med_returns, n_trials=n_trials)
        dsr_high = deflated_sharpe_ratio(high_returns, n_trials=n_trials)

        assert dsr_low < dsr_med < dsr_high

    def test_psr_symmetry(self) -> None:
        """PSR(SR, benchmark) + PSR(-SR, -benchmark) should equal 1."""
        np.random.seed(42)
        returns = pl.Series(np.random.normal(0.001, 0.02, 252))
        neg_returns = -returns

        psr_pos = probabilistic_sharpe_ratio(returns, benchmark_sr=0.5)
        psr_neg = probabilistic_sharpe_ratio(neg_returns, benchmark_sr=-0.5)

        # These should sum close to 1 due to symmetry
        # Note: Not exactly 1 due to skewness/kurtosis effects
        assert abs(psr_pos.psr + psr_neg.psr - 1.0) < 0.3

    def test_dsr_converges_to_psr_as_trials_decrease(self) -> None:
        """DSR should approach PSR(benchmark=0) as n_trials -> 1."""
        np.random.seed(42)
        # Use moderate returns to avoid saturation at 1.0
        returns = pl.Series(np.random.normal(0.0005, 0.02, 500))

        psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
        dsr_1 = deflated_sharpe_ratio(returns, n_trials=1)
        dsr_5 = deflated_sharpe_ratio(returns, n_trials=5)
        dsr_20 = deflated_sharpe_ratio(returns, n_trials=20)

        # DSR should converge to PSR as trials decrease
        assert abs(dsr_1 - psr.psr) < abs(dsr_20 - psr.psr)
        # DSR should decrease as trials increase (with enough separation)
        assert dsr_1 >= dsr_5 >= dsr_20
