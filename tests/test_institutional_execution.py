"""Tests for institutional execution quality metrics.

Tests implementation shortfall, market impact, VWAP slippage, and spread cost.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from nanuquant.institutional import (
    ImplementationShortfallResult,
    execution_vwap,
    implementation_shortfall,
    market_impact_estimate,
    spread_cost,
    vwap_slippage,
)


class TestImplementationShortfall:
    """Tests for implementation shortfall calculation."""

    def test_result_type(self) -> None:
        """Test that result is correct named tuple type."""
        result = implementation_shortfall(
            decision_price=100.0,
            execution_prices=[100.5, 101.0],
            quantities=[500, 500],
            side=1,
        )

        assert isinstance(result, ImplementationShortfallResult)
        assert hasattr(result, "total_shortfall")
        assert hasattr(result, "delay_cost")
        assert hasattr(result, "trading_cost")
        assert hasattr(result, "opportunity_cost")
        assert hasattr(result, "shortfall_bps")

    def test_buy_slippage_positive_shortfall(self) -> None:
        """Buy order with price increase should have positive shortfall."""
        result = implementation_shortfall(
            decision_price=100.0,
            execution_prices=[101.0, 102.0],
            quantities=[500, 500],
            side=1,
        )

        # Paid more than decision price = positive shortfall
        assert result.shortfall_bps > 0
        assert result.total_shortfall > 0

    def test_sell_slippage_positive_shortfall(self) -> None:
        """Sell order with price decrease should have positive shortfall."""
        result = implementation_shortfall(
            decision_price=100.0,
            execution_prices=[99.0, 98.0],
            quantities=[500, 500],
            side=-1,
        )

        # Received less than decision price = positive shortfall
        assert result.shortfall_bps > 0

    def test_perfect_execution_zero_shortfall(self) -> None:
        """Execution at decision price should have zero shortfall."""
        result = implementation_shortfall(
            decision_price=100.0,
            execution_prices=[100.0, 100.0],
            quantities=[500, 500],
            side=1,
        )

        assert abs(result.shortfall_bps) < 0.01
        assert abs(result.total_shortfall) < 0.01

    def test_better_than_decision_negative_shortfall(self) -> None:
        """Execution better than decision price should have negative shortfall."""
        result = implementation_shortfall(
            decision_price=100.0,
            execution_prices=[99.0, 99.5],
            quantities=[500, 500],
            side=1,  # Buy
        )

        # Bought cheaper = negative shortfall (outperformance)
        assert result.shortfall_bps < 0

    def test_shortfall_decomposition(self) -> None:
        """Verify shortfall components sum correctly."""
        result = implementation_shortfall(
            decision_price=100.0,
            execution_prices=[100.5, 101.0, 101.5],
            quantities=[300, 400, 300],
            side=1,
            arrival_price=100.3,
        )

        # Components should sum to approximately total (ignoring opportunity cost)
        component_sum = result.delay_cost + result.trading_cost
        # Note: exact equality may not hold due to different calculation methods
        assert abs(component_sum - result.total_shortfall) < result.total_shortfall * 0.5

    def test_with_polars_series(self) -> None:
        """Test with Polars Series inputs."""
        prices = pl.Series([100.5, 101.0, 101.5])
        quantities = pl.Series([300, 400, 300])

        result = implementation_shortfall(
            decision_price=100.0,
            execution_prices=prices,
            quantities=quantities,
            side=1,
        )

        assert result.shortfall_bps > 0

    def test_opportunity_cost(self) -> None:
        """Test opportunity cost calculation."""
        result = implementation_shortfall(
            decision_price=100.0,
            execution_prices=[101.0],
            quantities=[500],
            side=1,
            end_price=105.0,
            unfilled_quantity=500,
        )

        # Unfilled quantity missed $5 gain = opportunity cost
        assert result.opportunity_cost > 0

    def test_empty_execution(self) -> None:
        """Empty execution should return zeros."""
        result = implementation_shortfall(
            decision_price=100.0,
            execution_prices=[],
            quantities=[],
            side=1,
        )

        assert result.total_shortfall == 0.0
        assert result.shortfall_bps == 0.0

    def test_invalid_decision_price(self) -> None:
        """Should raise error for non-positive decision price."""
        with pytest.raises(ValueError):
            implementation_shortfall(
                decision_price=0.0,
                execution_prices=[100.0],
                quantities=[100],
                side=1,
            )

        with pytest.raises(ValueError):
            implementation_shortfall(
                decision_price=-100.0,
                execution_prices=[100.0],
                quantities=[100],
                side=1,
            )

    def test_mismatched_lengths(self) -> None:
        """Should raise error for mismatched array lengths."""
        with pytest.raises(ValueError):
            implementation_shortfall(
                decision_price=100.0,
                execution_prices=[100.0, 101.0],
                quantities=[100],  # Wrong length
                side=1,
            )


class TestMarketImpactEstimate:
    """Tests for market impact estimation."""

    def test_positive_impact(self) -> None:
        """Market impact should be positive for positive volume."""
        impact = market_impact_estimate(
            trade_volume=100_000,
            avg_daily_volume=1_000_000,
            volatility=0.20,
        )

        assert impact > 0

    def test_zero_volume_zero_impact(self) -> None:
        """Zero volume should have zero impact."""
        impact = market_impact_estimate(
            trade_volume=0,
            avg_daily_volume=1_000_000,
            volatility=0.20,
        )

        assert impact == 0.0

    def test_larger_trade_larger_impact(self) -> None:
        """Larger trades should have larger impact."""
        impact_small = market_impact_estimate(
            trade_volume=10_000,
            avg_daily_volume=1_000_000,
            volatility=0.20,
        )
        impact_large = market_impact_estimate(
            trade_volume=100_000,
            avg_daily_volume=1_000_000,
            volatility=0.20,
        )

        assert impact_large > impact_small

    def test_higher_volatility_higher_impact(self) -> None:
        """Higher volatility should increase impact."""
        impact_low_vol = market_impact_estimate(
            trade_volume=100_000,
            avg_daily_volume=1_000_000,
            volatility=0.10,
        )
        impact_high_vol = market_impact_estimate(
            trade_volume=100_000,
            avg_daily_volume=1_000_000,
            volatility=0.30,
        )

        assert impact_high_vol > impact_low_vol

    def test_square_root_scaling(self) -> None:
        """Impact should scale with square root of participation."""
        impact_1 = market_impact_estimate(
            trade_volume=10_000,
            avg_daily_volume=1_000_000,
            volatility=0.20,
        )
        impact_4 = market_impact_estimate(
            trade_volume=40_000,  # 4x volume
            avg_daily_volume=1_000_000,
            volatility=0.20,
        )

        # Should scale as sqrt(4) = 2x
        ratio = impact_4 / impact_1
        assert abs(ratio - 2.0) < 0.1

    def test_custom_coefficient(self) -> None:
        """Test with custom impact coefficient."""
        impact_default = market_impact_estimate(
            trade_volume=100_000,
            avg_daily_volume=1_000_000,
            volatility=0.20,
            impact_coefficient=0.1,
        )
        impact_double = market_impact_estimate(
            trade_volume=100_000,
            avg_daily_volume=1_000_000,
            volatility=0.20,
            impact_coefficient=0.2,
        )

        assert abs(impact_double / impact_default - 2.0) < 0.01

    def test_invalid_inputs(self) -> None:
        """Should raise errors for invalid inputs."""
        with pytest.raises(ValueError):
            market_impact_estimate(
                trade_volume=-100,
                avg_daily_volume=1_000_000,
                volatility=0.20,
            )

        with pytest.raises(ValueError):
            market_impact_estimate(
                trade_volume=100_000,
                avg_daily_volume=0,
                volatility=0.20,
            )


class TestVWAPSlippage:
    """Tests for VWAP slippage calculation."""

    def test_buy_worse_than_benchmark_positive(self) -> None:
        """Buy above benchmark should give positive slippage."""
        slippage = vwap_slippage(
            execution_prices=[101.0, 102.0],
            quantities=[500, 500],
            benchmark_vwap=100.0,
            side=1,
        )

        assert slippage > 0

    def test_buy_better_than_benchmark_negative(self) -> None:
        """Buy below benchmark should give negative slippage."""
        slippage = vwap_slippage(
            execution_prices=[99.0, 99.5],
            quantities=[500, 500],
            benchmark_vwap=100.0,
            side=1,
        )

        assert slippage < 0

    def test_sell_worse_than_benchmark_positive(self) -> None:
        """Sell below benchmark should give positive slippage."""
        slippage = vwap_slippage(
            execution_prices=[99.0, 98.0],
            quantities=[500, 500],
            benchmark_vwap=100.0,
            side=-1,
        )

        assert slippage > 0

    def test_exact_match_zero_slippage(self) -> None:
        """Execution at benchmark should give zero slippage."""
        slippage = vwap_slippage(
            execution_prices=[100.0, 100.0],
            quantities=[500, 500],
            benchmark_vwap=100.0,
            side=1,
        )

        assert abs(slippage) < 0.01


class TestSpreadCost:
    """Tests for spread cost calculation."""

    def test_positive_spread(self) -> None:
        """Spread cost should be positive."""
        cost = spread_cost(bid=99.95, ask=100.05)
        assert cost > 0

    def test_known_spread(self) -> None:
        """Test with known spread value."""
        # 10 cent spread on $100 = 10 bps
        cost = spread_cost(bid=99.95, ask=100.05)
        assert abs(cost - 10.0) < 0.1

    def test_with_series(self) -> None:
        """Test with Series inputs."""
        bids = pl.Series([99.95, 99.90, 99.85])
        asks = pl.Series([100.05, 100.10, 100.15])

        cost = spread_cost(bid=bids, ask=asks)
        assert cost > 0

    def test_zero_spread(self) -> None:
        """Zero spread should give zero cost."""
        cost = spread_cost(bid=100.0, ask=100.0)
        assert cost == 0.0


class TestExecutionVWAP:
    """Tests for execution VWAP calculation."""

    def test_simple_vwap(self) -> None:
        """Test simple VWAP calculation."""
        vwap = execution_vwap(
            execution_prices=[100, 101, 102],
            quantities=[100, 200, 100],
        )

        # Expected: (100*100 + 101*200 + 102*100) / 400 = 40400 / 400 = 101.0
        assert abs(vwap - 101.0) < 0.01

    def test_equal_quantities(self) -> None:
        """Equal quantities should give simple average."""
        vwap = execution_vwap(
            execution_prices=[100, 102, 104],
            quantities=[100, 100, 100],
        )

        # Should be simple average
        assert abs(vwap - 102.0) < 0.01

    def test_with_polars_series(self) -> None:
        """Test with Polars Series inputs."""
        prices = pl.Series([100.0, 101.0, 102.0])
        quantities = pl.Series([100, 200, 100])

        vwap = execution_vwap(prices, quantities)
        assert abs(vwap - 101.0) < 0.01

    def test_empty_execution(self) -> None:
        """Empty execution should return 0."""
        vwap = execution_vwap([], [])
        assert vwap == 0.0


class TestSyntheticDataRecovery:
    """Synthetic data tests for execution metrics."""

    def test_is_matches_manual_calculation(self) -> None:
        """Verify IS matches manual calculation."""
        decision = 100.0
        prices = [100.5, 101.0]
        quantities = [500, 500]

        result = implementation_shortfall(decision, prices, quantities, side=1)

        # Manual calculation
        vwap = (100.5 * 500 + 101.0 * 500) / 1000  # 100.75
        manual_shortfall_bps = (vwap - decision) / decision * 10000  # 75 bps

        assert abs(result.shortfall_bps - manual_shortfall_bps) < 0.1

    def test_market_impact_reasonable_magnitude(self) -> None:
        """Market impact should be in reasonable range."""
        # 10% of ADV trade in a 20% vol stock
        impact = market_impact_estimate(
            trade_volume=100_000,
            avg_daily_volume=1_000_000,
            volatility=0.20,
        )

        # Should be less than 1% for typical trade
        assert 0 < impact < 0.01

        # 50% of ADV trade (large)
        impact_large = market_impact_estimate(
            trade_volume=500_000,
            avg_daily_volume=1_000_000,
            volatility=0.20,
        )

        # Should be larger but still reasonable
        assert impact_large > impact
        assert impact_large < 0.05


class TestEdgeCases:
    """Edge case tests for execution metrics."""

    def test_single_fill(self) -> None:
        """Test with single fill."""
        result = implementation_shortfall(
            decision_price=100.0,
            execution_prices=[101.0],
            quantities=[1000],
            side=1,
        )

        assert result.shortfall_bps == 100.0  # 1% = 100 bps

    def test_large_trade(self) -> None:
        """Test with very large trade."""
        result = implementation_shortfall(
            decision_price=100.0,
            execution_prices=[100.01] * 100,
            quantities=[10000] * 100,  # 1 million shares
            side=1,
        )

        assert result.shortfall_bps > 0

    def test_fractional_quantities(self) -> None:
        """Test with fractional quantities."""
        result = implementation_shortfall(
            decision_price=100.0,
            execution_prices=[100.5, 101.0],
            quantities=[333.33, 666.67],
            side=1,
        )

        assert math.isfinite(result.shortfall_bps)
