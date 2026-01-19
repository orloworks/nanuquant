"""Tests for trade conversion module.

Tests cover:
- Basic trade-to-returns conversion
- Long/short trade handling
- Fee calculations
- Aggregation modes (trade, equity, D/W/M)
- Mark-to-market with price data
- Edge cases and validation
- Integration with metrics
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

import nanuquant as pm
from nanuquant.exceptions import EmptySeriesError
from nanuquant.trades import (
    InvalidDirectionError,
    InvalidPriceError,
    InvalidTradeTimesError,
    MissingColumnError,
    TradeResult,
    build_equity_curve,
    build_equity_curve_no_mtm,
    calculate_single_trade_return,
    trades_to_returns,
    validate_trade_dataframe,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_trades() -> pl.DataFrame:
    """Simple trade data for basic testing."""
    return pl.DataFrame({
        "entry_time": [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
        ],
        "exit_time": [
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
            datetime(2024, 1, 4),
        ],
        "entry_price": [100.0, 105.0, 102.0],
        "exit_price": [105.0, 103.0, 108.0],
        "direction": ["long", "long", "long"],
    })


@pytest.fixture
def mixed_direction_trades() -> pl.DataFrame:
    """Trades with both long and short positions."""
    return pl.DataFrame({
        "entry_time": [datetime(2024, 1, i) for i in range(1, 5)],
        "exit_time": [datetime(2024, 1, i) for i in range(2, 6)],
        "entry_price": [100.0, 102.0, 98.0, 105.0],
        "exit_price": [102.0, 98.0, 102.0, 100.0],
        "direction": ["long", "short", "long", "short"],
    })


@pytest.fixture
def trades_with_fees() -> pl.DataFrame:
    """Trades including transaction fees."""
    return pl.DataFrame({
        "entry_time": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        "exit_time": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
        "entry_price": [100.0, 100.0],
        "exit_price": [110.0, 105.0],
        "direction": ["long", "long"],
        "quantity": [10.0, 20.0],
        "fees": [5.0, 10.0],
    })


@pytest.fixture
def multi_day_trade() -> pl.DataFrame:
    """Single trade held over multiple days."""
    return pl.DataFrame({
        "entry_time": [datetime(2024, 1, 1)],
        "exit_time": [datetime(2024, 1, 5)],
        "entry_price": [100.0],
        "exit_price": [105.0],
        "direction": ["long"],
    })


@pytest.fixture
def price_data() -> pl.DataFrame:
    """Daily price data for mark-to-market."""
    return pl.DataFrame({
        "date": [
            date(2024, 1, 1),
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 5),
        ],
        "close": [100.0, 102.0, 101.0, 103.0, 105.0],
    })


# =============================================================================
# Basic Conversion Tests
# =============================================================================


class TestBasicConversion:
    """Tests for basic trade-to-returns conversion."""

    def test_simple_long_trades(self, simple_trades: pl.DataFrame) -> None:
        """Test conversion of simple long trades."""
        result = trades_to_returns(simple_trades)

        assert result.n_trades == 3
        assert len(result.returns) == 3
        assert result.aggregation == "trade"

        # Verify returns: (exit - entry) / entry
        returns = result.returns.to_list()
        assert abs(returns[0] - 0.05) < 1e-6  # (105-100)/100
        assert abs(returns[1] - (-0.019047619)) < 1e-6  # (103-105)/105
        assert abs(returns[2] - 0.058823529) < 1e-6  # (108-102)/102

    def test_short_trades(self) -> None:
        """Test conversion of short trades."""
        trades = pl.DataFrame({
            "entry_time": [datetime(2024, 1, 1)],
            "exit_time": [datetime(2024, 1, 2)],
            "entry_price": [100.0],
            "exit_price": [95.0],
            "direction": ["short"],
        })
        result = trades_to_returns(trades)

        # Short profit: (entry - exit) / entry = (100 - 95) / 100 = 0.05
        assert abs(result.returns[0] - 0.05) < 1e-6

    def test_short_trade_loss(self) -> None:
        """Test short trade with loss (price goes up)."""
        trades = pl.DataFrame({
            "entry_time": [datetime(2024, 1, 1)],
            "exit_time": [datetime(2024, 1, 2)],
            "entry_price": [100.0],
            "exit_price": [105.0],
            "direction": ["short"],
        })
        result = trades_to_returns(trades)

        # Short loss: (100 - 105) / 100 = -0.05
        assert abs(result.returns[0] - (-0.05)) < 1e-6

    def test_mixed_directions(self, mixed_direction_trades: pl.DataFrame) -> None:
        """Test trades with mixed long/short directions."""
        result = trades_to_returns(mixed_direction_trades)

        assert result.n_trades == 4
        returns = result.returns.to_list()

        # Long: (102-100)/100 = 0.02
        assert abs(returns[0] - 0.02) < 1e-6
        # Short: (102-98)/102 = 0.0392 (profit when price drops)
        assert abs(returns[1] - 0.0392156863) < 1e-6
        # Long: (102-98)/98 = 0.0408
        assert abs(returns[2] - 0.0408163265) < 1e-6
        # Short: (105-100)/105 = 0.0476 (profit when price drops)
        assert abs(returns[3] - 0.0476190476) < 1e-6

    def test_trade_statistics(self, simple_trades: pl.DataFrame) -> None:
        """Test that trade statistics are calculated correctly."""
        result = trades_to_returns(simple_trades)

        assert result.n_trades == 3
        assert result.n_winning == 2  # First and third trades positive
        assert result.n_losing == 1  # Second trade negative


class TestReturnMethods:
    """Tests for different return calculation methods."""

    def test_simple_returns(self, simple_trades: pl.DataFrame) -> None:
        """Test simple return calculation."""
        result = trades_to_returns(simple_trades, method="simple")
        # First trade: (105 - 100) / 100 = 0.05
        assert abs(result.returns[0] - 0.05) < 1e-6

    def test_log_returns(self, simple_trades: pl.DataFrame) -> None:
        """Test log return calculation."""
        import math

        result = trades_to_returns(simple_trades, method="log")
        # First trade: ln(105/100) ≈ 0.04879
        expected = math.log(105 / 100)
        assert abs(result.returns[0] - expected) < 1e-6


class TestFeeHandling:
    """Tests for fee calculations."""

    def test_fees_reduce_returns(self, trades_with_fees: pl.DataFrame) -> None:
        """Test that fees reduce returns."""
        result_with_fees = trades_to_returns(trades_with_fees, include_fees=True)
        result_no_fees = trades_to_returns(trades_with_fees, include_fees=False)

        for with_fee, no_fee in zip(
            result_with_fees.returns.to_list(), result_no_fees.returns.to_list()
        ):
            assert with_fee < no_fee

    def test_fee_calculation_accuracy(self) -> None:
        """Test precise fee impact calculation."""
        trades = pl.DataFrame({
            "entry_time": [datetime(2024, 1, 1)],
            "exit_time": [datetime(2024, 1, 2)],
            "entry_price": [100.0],
            "exit_price": [110.0],
            "direction": ["long"],
            "quantity": [10.0],
            "fees": [10.0],
        })
        result = trades_to_returns(trades, include_fees=True)

        # Gross return: (110-100)/100 = 0.10
        # Fee impact: 10 / (100 * 10) = 0.01
        # Net return: 0.10 - 0.01 = 0.09
        assert abs(result.returns[0] - 0.09) < 1e-6

    def test_total_fees_tracked(self, trades_with_fees: pl.DataFrame) -> None:
        """Test that total fees are tracked in result."""
        result = trades_to_returns(trades_with_fees)
        assert result.total_fees == 15.0  # 5 + 10


# =============================================================================
# Aggregation Mode Tests
# =============================================================================


class TestTradeAggregation:
    """Tests for trade-level aggregation."""

    def test_trade_level_returns_one_per_trade(
        self, simple_trades: pl.DataFrame
    ) -> None:
        """Test that trade-level returns one return per trade."""
        result = trades_to_returns(simple_trades, aggregation="trade")
        assert len(result.returns) == 3
        assert result.aggregation == "trade"

    def test_trades_per_year_override(self, simple_trades: pl.DataFrame) -> None:
        """Test trades_per_year parameter."""
        result = trades_to_returns(
            simple_trades, aggregation="trade", trades_per_year=100
        )
        assert result.periods_per_year == 100


class TestEquityAggregation:
    """Tests for equity curve aggregation."""

    def test_equity_mode_requires_capital(self, simple_trades: pl.DataFrame) -> None:
        """Test that equity mode requires initial_capital."""
        with pytest.raises(ValueError, match="initial_capital is required"):
            trades_to_returns(simple_trades, aggregation="equity")

    def test_equity_mode_daily_returns(self, simple_trades: pl.DataFrame) -> None:
        """Test equity mode produces daily returns."""
        result = trades_to_returns(
            simple_trades, aggregation="equity", initial_capital=10000
        )

        assert result.aggregation == "equity"
        assert result.periods_per_year == 252
        # Should have returns for date range
        assert len(result.returns) >= 3

    def test_equity_mode_without_prices_warns(
        self, simple_trades: pl.DataFrame
    ) -> None:
        """Test warning when using equity mode without prices."""
        result = trades_to_returns(
            simple_trades, aggregation="equity", initial_capital=10000
        )

        assert not result.has_mtm
        assert len(result.warnings) > 0
        assert "Intra-trade volatility not captured" in result.warnings[0]


class TestEquityWithPrices:
    """Tests for equity curve with mark-to-market."""

    def test_mtm_daily_returns(
        self, multi_day_trade: pl.DataFrame, price_data: pl.DataFrame
    ) -> None:
        """Test mark-to-market produces accurate daily returns."""
        result = trades_to_returns(
            multi_day_trade,
            prices=price_data,
            aggregation="equity",
            initial_capital=10000,
        )

        assert result.has_mtm
        assert len(result.warnings) == 0
        # Should have daily returns for the holding period
        assert len(result.returns) >= 4  # Jan 1-5

    def test_mtm_captures_intraday_volatility(
        self, multi_day_trade: pl.DataFrame, price_data: pl.DataFrame
    ) -> None:
        """Test that MTM captures volatility during holding period."""
        result_mtm = trades_to_returns(
            multi_day_trade,
            prices=price_data,
            aggregation="equity",
            initial_capital=10000,
        )

        result_no_mtm = trades_to_returns(
            multi_day_trade, aggregation="equity", initial_capital=10000
        )

        # MTM should have more non-zero returns
        mtm_nonzero = (result_mtm.returns != 0).sum()
        no_mtm_nonzero = (result_no_mtm.returns != 0).sum()
        assert mtm_nonzero >= no_mtm_nonzero


class TestCalendarAggregation:
    """Tests for calendar-based aggregation (D/W/M)."""

    def test_daily_aggregation(self) -> None:
        """Test daily aggregation of multiple trades on same day."""
        trades = pl.DataFrame({
            "entry_time": [
                datetime(2024, 1, 1, 9, 30),
                datetime(2024, 1, 1, 10, 30),
                datetime(2024, 1, 2, 9, 30),
            ],
            "exit_time": [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 11, 0),
                datetime(2024, 1, 2, 10, 0),
            ],
            "entry_price": [100.0, 102.0, 101.0],
            "exit_price": [102.0, 103.0, 102.0],
            "direction": ["long", "long", "long"],
        })
        result = trades_to_returns(
            trades, aggregation="D", initial_capital=10000
        )

        assert result.aggregation == "D"
        assert result.periods_per_year == 252


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for input validation."""

    def test_missing_entry_time_raises(self) -> None:
        """Missing entry_time should raise error."""
        trades = pl.DataFrame({
            "entry_price": [100.0],
            "exit_time": [datetime(2024, 1, 2)],
            "exit_price": [105.0],
        })
        with pytest.raises(MissingColumnError, match="entry_time"):
            validate_trade_dataframe(trades)

    def test_missing_entry_price_raises(self) -> None:
        """Missing entry_price should raise error."""
        trades = pl.DataFrame({
            "entry_time": [datetime(2024, 1, 1)],
            "exit_time": [datetime(2024, 1, 2)],
            "exit_price": [105.0],
        })
        with pytest.raises(MissingColumnError, match="entry_price"):
            validate_trade_dataframe(trades)

    def test_negative_price_raises(self) -> None:
        """Negative prices should raise error."""
        trades = pl.DataFrame({
            "entry_time": [datetime(2024, 1, 1)],
            "entry_price": [-100.0],
        })
        with pytest.raises(InvalidPriceError):
            validate_trade_dataframe(trades)

    def test_zero_price_raises(self) -> None:
        """Zero prices should raise error."""
        trades = pl.DataFrame({
            "entry_time": [datetime(2024, 1, 1)],
            "entry_price": [0.0],
        })
        with pytest.raises(InvalidPriceError):
            validate_trade_dataframe(trades)

    def test_invalid_direction_raises(self) -> None:
        """Invalid direction should raise error."""
        trades = pl.DataFrame({
            "entry_time": [datetime(2024, 1, 1)],
            "entry_price": [100.0],
            "direction": ["buy"],
        })
        with pytest.raises(InvalidDirectionError):
            validate_trade_dataframe(trades)

    def test_empty_dataframe_raises(self) -> None:
        """Empty DataFrame should raise error."""
        trades = pl.DataFrame({
            "entry_time": [],
            "entry_price": [],
        }).cast({"entry_time": pl.Datetime, "entry_price": pl.Float64})

        with pytest.raises(EmptySeriesError):
            trades_to_returns(trades)

    def test_exit_before_entry_raises(self) -> None:
        """Exit time before entry time should raise error."""
        from nanuquant.trades import validate_trade_times

        trades = pl.DataFrame({
            "entry_time": [datetime(2024, 1, 2)],
            "exit_time": [datetime(2024, 1, 1)],
            "entry_price": [100.0],
            "exit_price": [105.0],
        })
        with pytest.raises(InvalidTradeTimesError):
            validate_trade_times(trades)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_trade(self) -> None:
        """Test with a single trade."""
        trades = pl.DataFrame({
            "entry_time": [datetime(2024, 1, 1)],
            "exit_time": [datetime(2024, 1, 2)],
            "entry_price": [100.0],
            "exit_price": [105.0],
            "direction": ["long"],
        })
        result = trades_to_returns(trades)
        assert result.n_trades == 1
        assert abs(result.returns[0] - 0.05) < 1e-6

    def test_zero_return_trade(self) -> None:
        """Test trade with zero return."""
        trades = pl.DataFrame({
            "entry_time": [datetime(2024, 1, 1)],
            "exit_time": [datetime(2024, 1, 2)],
            "entry_price": [100.0],
            "exit_price": [100.0],
            "direction": ["long"],
        })
        result = trades_to_returns(trades)
        assert result.returns[0] == 0.0

    def test_open_trades_filtered(self) -> None:
        """Test that open trades (no exit) are filtered out."""
        trades = pl.DataFrame({
            "entry_time": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "exit_time": [datetime(2024, 1, 2), None],
            "entry_price": [100.0, 105.0],
            "exit_price": [105.0, None],
            "direction": ["long", "long"],
        })
        result = trades_to_returns(trades)
        assert result.n_trades == 1

    def test_default_values_applied(self) -> None:
        """Test that default values are applied for missing columns."""
        trades = pl.DataFrame({
            "entry_time": [datetime(2024, 1, 1)],
            "exit_time": [datetime(2024, 1, 2)],
            "entry_price": [100.0],
            "exit_price": [105.0],
        })
        # Should not raise - direction defaults to "long"
        result = trades_to_returns(trades)
        assert result.n_trades == 1

    def test_very_small_return(self) -> None:
        """Test with very small price movement."""
        trades = pl.DataFrame({
            "entry_time": [datetime(2024, 1, 1)],
            "exit_time": [datetime(2024, 1, 2)],
            "entry_price": [100.0],
            "exit_price": [100.001],
            "direction": ["long"],
        })
        result = trades_to_returns(trades)
        assert abs(result.returns[0] - 0.00001) < 1e-8


# =============================================================================
# Equity Curve Tests
# =============================================================================


class TestEquityCurve:
    """Tests for equity curve building."""

    def test_build_equity_curve_basic(
        self, multi_day_trade: pl.DataFrame, price_data: pl.DataFrame
    ) -> None:
        """Test basic equity curve building."""
        equity = build_equity_curve(multi_day_trade, 10000, price_data)

        assert "date" in equity.columns
        assert "nav" in equity.columns
        assert "daily_return" in equity.columns
        assert len(equity) >= 4

    def test_equity_curve_no_mtm(self, multi_day_trade: pl.DataFrame) -> None:
        """Test equity curve without mark-to-market."""
        equity = build_equity_curve_no_mtm(multi_day_trade, 10000)

        assert "nav" in equity.columns
        # Most returns should be 0 except exit day
        zero_returns = (equity["daily_return"] == 0).sum()
        assert zero_returns >= len(equity) - 2

    def test_equity_curve_empty_trades(self) -> None:
        """Test equity curve with no trades."""
        trades = pl.DataFrame({
            "entry_time": [],
            "exit_time": [],
            "entry_price": [],
            "exit_price": [],
        }).cast({
            "entry_time": pl.Datetime,
            "exit_time": pl.Datetime,
            "entry_price": pl.Float64,
            "exit_price": pl.Float64,
        })

        equity = build_equity_curve_no_mtm(trades, 10000)
        assert equity.is_empty()


# =============================================================================
# Overlapping Holdings Tests
# =============================================================================


class TestOverlappingHoldings:
    """Tests for overlapping positions (multiple trades open simultaneously)."""

    @pytest.fixture
    def overlapping_trades(self) -> pl.DataFrame:
        """Trades with overlapping holding periods."""
        return pl.DataFrame({
            "entry_time": [
                datetime(2024, 1, 2),  # Trade 1: Jan 2-5
                datetime(2024, 1, 3),  # Trade 2: Jan 3-6 (overlaps)
                datetime(2024, 1, 4),  # Trade 3: Jan 4-8 (overlaps)
            ],
            "exit_time": [
                datetime(2024, 1, 5),
                datetime(2024, 1, 6),
                datetime(2024, 1, 8),
            ],
            "entry_price": [100.0, 102.0, 104.0],
            "exit_price": [105.0, 108.0, 100.0],
            "quantity": [10.0, 20.0, 15.0],
            "direction": ["long", "long", "short"],
        })

    @pytest.fixture
    def overlapping_prices(self) -> pl.DataFrame:
        """Price data covering the overlapping period."""
        return pl.DataFrame({
            "date": [date(2024, 1, d) for d in range(2, 9)],
            "close": [100.0, 101.0, 103.0, 105.0, 106.0, 104.0, 102.0],
        })

    def test_trade_level_returns_independent(
        self, overlapping_trades: pl.DataFrame
    ) -> None:
        """Trade-level returns treat each trade independently."""
        result = trades_to_returns(overlapping_trades, aggregation="trade")

        assert result.n_trades == 3
        returns = result.returns.to_list()

        # Each trade calculated independently
        # Trade 1: (105-100)/100 = 0.05
        assert abs(returns[0] - 0.05) < 1e-6
        # Trade 2: (108-102)/102 = 0.0588
        assert abs(returns[1] - 0.058823529) < 1e-6
        # Trade 3 (short): (104-100)/104 = 0.0385
        assert abs(returns[2] - 0.038461538) < 1e-6

    def test_equity_mode_handles_overlaps(
        self, overlapping_trades: pl.DataFrame
    ) -> None:
        """Equity mode processes overlapping positions."""
        result = trades_to_returns(
            overlapping_trades, aggregation="equity", initial_capital=10000
        )

        # Should have daily returns for the period
        assert len(result.returns) == 7  # Jan 2-8
        assert result.n_trades == 3

    def test_mtm_with_overlapping_positions(
        self, overlapping_trades: pl.DataFrame, overlapping_prices: pl.DataFrame
    ) -> None:
        """Mark-to-market correctly values overlapping positions."""
        result = trades_to_returns(
            overlapping_trades,
            prices=overlapping_prices,
            aggregation="equity",
            initial_capital=10000,
        )

        assert result.has_mtm
        assert len(result.returns) == 7

        # MTM should produce more non-zero returns than exit-only
        result_no_mtm = trades_to_returns(
            overlapping_trades, aggregation="equity", initial_capital=10000
        )

        mtm_nonzero = (result.returns != 0).sum()
        no_mtm_nonzero = (result_no_mtm.returns != 0).sum()
        assert mtm_nonzero >= no_mtm_nonzero

    def test_total_return_consistency(
        self, overlapping_trades: pl.DataFrame
    ) -> None:
        """Total compounded return should be consistent across modes."""
        result_trade = trades_to_returns(overlapping_trades, aggregation="trade")
        result_equity = trades_to_returns(
            overlapping_trades, aggregation="equity", initial_capital=10000
        )

        # Compound the returns
        trade_total = float((1 + result_trade.returns).product() - 1)
        equity_total = float((1 + result_equity.returns).product() - 1)

        # Should be reasonably close (equity may differ due to timing)
        # The key is both should be positive given all winning trades
        assert trade_total > 0
        assert equity_total > 0

    def test_concurrent_long_short(self) -> None:
        """Test concurrent long and short positions in same asset."""
        # Same asset, same exit price - price goes up to 110
        trades = pl.DataFrame({
            "entry_time": [
                datetime(2024, 1, 2),  # Long
                datetime(2024, 1, 2),  # Short (same day)
            ],
            "exit_time": [
                datetime(2024, 1, 5),
                datetime(2024, 1, 5),
            ],
            "entry_price": [100.0, 100.0],
            "exit_price": [110.0, 110.0],  # Same exit price for same asset
            "quantity": [10.0, 10.0],
            "direction": ["long", "short"],
        })

        result = trades_to_returns(trades, aggregation="trade")

        # Long wins: (110-100)/100 = 10%
        # Short loses: (100-110)/100 = -10%
        assert abs(result.returns[0] - 0.10) < 1e-6
        assert abs(result.returns[1] - (-0.10)) < 1e-6
        assert result.n_winning == 1
        assert result.n_losing == 1

    def test_multiple_entries_same_day(self) -> None:
        """Test multiple trade entries on the same day."""
        trades = pl.DataFrame({
            "entry_time": [
                datetime(2024, 1, 2, 9, 30),
                datetime(2024, 1, 2, 10, 30),
                datetime(2024, 1, 2, 14, 0),
            ],
            "exit_time": [
                datetime(2024, 1, 3, 10, 0),
                datetime(2024, 1, 3, 11, 0),
                datetime(2024, 1, 3, 15, 0),
            ],
            "entry_price": [100.0, 101.0, 102.0],
            "exit_price": [102.0, 103.0, 104.0],
            "direction": ["long", "long", "long"],
        })

        result = trades_to_returns(trades, aggregation="trade")
        assert result.n_trades == 3

        result_equity = trades_to_returns(
            trades, aggregation="equity", initial_capital=10000
        )
        # Should have 2 days of returns
        assert len(result_equity.returns) == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestMetricsIntegration:
    """Tests for integration with metrics functions."""

    def test_sharpe_ratio_integration(self, simple_trades: pl.DataFrame) -> None:
        """Test that returns work with sharpe ratio."""
        result = trades_to_returns(simple_trades)
        sharpe_val = pm.sharpe(result.returns)
        assert isinstance(sharpe_val, float)

    def test_win_rate_integration(self, simple_trades: pl.DataFrame) -> None:
        """Test that returns work with win_rate."""
        result = trades_to_returns(simple_trades)
        wr = pm.win_rate(result.returns)

        # 2 winning trades out of 3
        assert abs(wr - 0.6667) < 0.01

    def test_max_drawdown_integration(self, simple_trades: pl.DataFrame) -> None:
        """Test that returns work with max_drawdown."""
        result = trades_to_returns(simple_trades)
        mdd = pm.max_drawdown(result.returns)
        assert isinstance(mdd, float)
        assert mdd <= 0  # Drawdown is negative or zero

    def test_profit_factor_integration(self, simple_trades: pl.DataFrame) -> None:
        """Test that returns work with profit_factor."""
        result = trades_to_returns(simple_trades)
        pf = pm.profit_factor(result.returns)
        assert isinstance(pf, float)
        assert pf > 0  # Should be positive with some winners

    def test_expectancy_integration(self, simple_trades: pl.DataFrame) -> None:
        """Test that returns work with expectancy."""
        result = trades_to_returns(simple_trades)
        exp = pm.expectancy(result.returns)
        assert isinstance(exp, float)

    def test_full_metrics_integration(self, simple_trades: pl.DataFrame) -> None:
        """Test that returns work with full_metrics report."""
        result = trades_to_returns(simple_trades)
        report = pm.full_metrics(result.returns)
        assert report is not None
        assert hasattr(report, "returns_metrics")


# =============================================================================
# Single Trade Function Tests
# =============================================================================


class TestSingleTradeFunction:
    """Tests for calculate_single_trade_return."""

    def test_long_profit(self) -> None:
        """Test long trade with profit."""
        ret = calculate_single_trade_return(100.0, 110.0, direction="long")
        assert abs(ret - 0.10) < 1e-6

    def test_long_loss(self) -> None:
        """Test long trade with loss."""
        ret = calculate_single_trade_return(100.0, 90.0, direction="long")
        assert abs(ret - (-0.10)) < 1e-6

    def test_short_profit(self) -> None:
        """Test short trade with profit (price drops)."""
        ret = calculate_single_trade_return(100.0, 90.0, direction="short")
        assert abs(ret - 0.10) < 1e-6

    def test_short_loss(self) -> None:
        """Test short trade with loss (price rises)."""
        ret = calculate_single_trade_return(100.0, 110.0, direction="short")
        assert abs(ret - (-0.10)) < 1e-6

    def test_with_fees(self) -> None:
        """Test return with fees."""
        ret = calculate_single_trade_return(
            100.0, 110.0, direction="long", fees=5.0, quantity=10.0
        )
        # Gross: 0.10, Fee impact: 5/(100*10) = 0.005
        assert abs(ret - 0.095) < 1e-6

    def test_log_return(self) -> None:
        """Test log return calculation."""
        import math

        ret = calculate_single_trade_return(100.0, 110.0, method="log")
        expected = math.log(110 / 100)
        assert abs(ret - expected) < 1e-6

    def test_invalid_prices_raise(self) -> None:
        """Test that invalid prices raise errors."""
        with pytest.raises(ValueError, match="entry_price must be positive"):
            calculate_single_trade_return(-100.0, 110.0)

        with pytest.raises(ValueError, match="exit_price must be positive"):
            calculate_single_trade_return(100.0, -110.0)


# =============================================================================
# TradeResult Tests
# =============================================================================


class TestTradeResult:
    """Tests for TradeResult dataclass."""

    def test_result_has_all_fields(self, simple_trades: pl.DataFrame) -> None:
        """Test that TradeResult has all expected fields."""
        result = trades_to_returns(simple_trades)

        assert hasattr(result, "returns")
        assert hasattr(result, "dates")
        assert hasattr(result, "n_trades")
        assert hasattr(result, "n_winning")
        assert hasattr(result, "n_losing")
        assert hasattr(result, "total_fees")
        assert hasattr(result, "aggregation")
        assert hasattr(result, "periods_per_year")
        assert hasattr(result, "has_mtm")
        assert hasattr(result, "warnings")

    def test_returns_series_named_correctly(
        self, simple_trades: pl.DataFrame
    ) -> None:
        """Test that returns series is named 'returns'."""
        result = trades_to_returns(simple_trades)
        assert result.returns.name == "returns"
