"""Integration tests using real market data (SPY, QQQ, BND).

These tests verify metrics against recognizable real-world data.
Data is pre-cached in tests/.data_cache/ - no network required.

Available data:
- SPY: 1993-present (~8000 days, 33 years)
- QQQ: 1999-present (~6700 days, 27 years)
- BND: 2007-present (~4700 days, 19 years)

Run with: pytest -m integration
Skip with: pytest -m "not integration"
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
import quantstats_lumi as qs

# Mark all tests as integration (real market data, no network needed - data is cached)
pytestmark = pytest.mark.integration


class TestRealDataSanityChecks:
    """Sanity checks that real data loaded correctly."""

    def test_spy_full_history(self, spy_returns_full: pd.Series) -> None:
        """SPY full history should have 30+ years of data."""
        assert len(spy_returns_full) > 7500  # ~30 years

    def test_qqq_full_history(self, qqq_returns_full: pd.Series) -> None:
        """QQQ full history should have 25+ years of data."""
        assert len(qqq_returns_full) > 6000  # ~25 years

    def test_bnd_full_history(self, bnd_returns_full: pd.Series) -> None:
        """BND full history should have 17+ years of data."""
        assert len(bnd_returns_full) > 4000  # ~17 years

    def test_spy_has_data(self, spy_returns: pd.Series) -> None:
        """SPY should have ~1250 trading days for 5 years."""
        assert len(spy_returns) > 1000
        assert len(spy_returns) < 1400

    def test_qqq_has_data(self, qqq_returns: pd.Series) -> None:
        """QQQ should have similar length to SPY."""
        assert len(qqq_returns) > 1000

    def test_bnd_has_data(self, bnd_returns: pd.Series) -> None:
        """BND should have similar length to SPY."""
        assert len(bnd_returns) > 1000

    def test_spy_reasonable_returns(self, spy_returns: pd.Series) -> None:
        """SPY daily returns should be in reasonable range."""
        assert spy_returns.min() > -0.15  # No daily drop > 15%
        assert spy_returns.max() < 0.15   # No daily gain > 15%
        assert abs(spy_returns.mean()) < 0.01  # Mean close to 0

    def test_qqq_more_volatile_than_bnd(
        self,
        qqq_returns: pd.Series,
        bnd_returns: pd.Series,
    ) -> None:
        """QQQ should be more volatile than BND."""
        qqq_vol = qqq_returns.std()
        bnd_vol = bnd_returns.std()
        assert qqq_vol > bnd_vol * 2  # QQQ at least 2x more volatile


class TestDifferentialVsQuantStats:
    """Compare our metrics against QuantStats using real market data."""

    # Tolerance for floating point comparison
    RTOL = 1e-8
    ATOL = 1e-12

    def test_spy_sharpe_vs_quantstats(
        self,
        spy_returns: pd.Series,
        spy_polars: pl.Series,
    ) -> None:
        """SPY Sharpe ratio should match QuantStats."""
        # TODO: Implement pm.sharpe and uncomment
        # expected = qs.stats.sharpe(spy_returns)
        # actual = pm.sharpe(spy_polars)
        # assert abs(actual - expected) < self.ATOL
        pass  # Placeholder until metrics implemented

    def test_spy_sortino_vs_quantstats(
        self,
        spy_returns: pd.Series,
        spy_polars: pl.Series,
    ) -> None:
        """SPY Sortino ratio should match QuantStats."""
        # TODO: Implement pm.sortino and uncomment
        pass

    def test_spy_max_drawdown_vs_quantstats(
        self,
        spy_returns: pd.Series,
        spy_polars: pl.Series,
    ) -> None:
        """SPY max drawdown should match QuantStats."""
        # TODO: Implement pm.max_drawdown and uncomment
        pass

    def test_spy_cagr_vs_quantstats(
        self,
        spy_returns: pd.Series,
        spy_polars: pl.Series,
    ) -> None:
        """SPY CAGR should match QuantStats."""
        # TODO: Implement pm.cagr and uncomment
        pass


class TestExpectedMarketBehavior:
    """Tests that verify expected market behavior with real data.

    These tests encode financial intuition and serve as sanity checks.
    """

    def test_spy_positive_long_term_return(self, spy_returns: pd.Series) -> None:
        """SPY should have positive total return over 5 years."""
        total_return = (1 + spy_returns).prod() - 1
        assert total_return > 0, "SPY should be positive over 5 years"

    def test_qqq_higher_return_than_bnd(
        self,
        qqq_returns: pd.Series,
        bnd_returns: pd.Series,
    ) -> None:
        """QQQ should outperform BND over 5 years (higher risk = higher return)."""
        qqq_total = (1 + qqq_returns).prod() - 1
        bnd_total = (1 + bnd_returns).prod() - 1
        assert qqq_total > bnd_total

    def test_spy_qqq_positive_correlation(
        self,
        spy_returns: pd.Series,
        qqq_returns: pd.Series,
    ) -> None:
        """SPY and QQQ should be highly correlated (both US equities)."""
        # Align on common dates
        df = pd.DataFrame({"SPY": spy_returns, "QQQ": qqq_returns}).dropna()
        corr = df["SPY"].corr(df["QQQ"])
        assert corr > 0.8, f"SPY/QQQ correlation should be > 0.8, got {corr}"

    def test_bnd_spy_low_correlation(
        self,
        spy_returns: pd.Series,
        bnd_returns: pd.Series,
    ) -> None:
        """BND and SPY should have low or negative correlation."""
        df = pd.DataFrame({"SPY": spy_returns, "BND": bnd_returns}).dropna()
        corr = df["SPY"].corr(df["BND"])
        assert corr < 0.5, f"SPY/BND correlation should be < 0.5, got {corr}"

    def test_spy_has_drawdowns(self, spy_returns: pd.Series) -> None:
        """SPY should have meaningful drawdowns (not always up)."""
        # Calculate drawdown series
        cumulative = (1 + spy_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        assert max_dd < -0.10, "SPY should have at least 10% drawdown over 5 years"

    def test_qqq_deeper_drawdown_than_bnd(
        self,
        qqq_returns: pd.Series,
        bnd_returns: pd.Series,
    ) -> None:
        """QQQ should have deeper max drawdown than BND."""

        def max_drawdown(returns: pd.Series) -> float:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()

        qqq_dd = max_drawdown(qqq_returns)
        bnd_dd = max_drawdown(bnd_returns)

        assert qqq_dd < bnd_dd, "QQQ should have deeper drawdown than BND"


class TestMultiAssetPortfolio:
    """Tests using the combined market data DataFrame."""

    def test_market_data_aligned(self, market_data_df: pl.DataFrame) -> None:
        """All three ETFs should be aligned with no nulls."""
        assert market_data_df.null_count().sum_horizontal()[0] == 0
        assert market_data_df.shape[1] == 3  # SPY, QQQ, BND
        assert market_data_df.shape[0] > 1000  # At least 1000 aligned days

    def test_equal_weight_portfolio(self, market_data_df: pl.DataFrame) -> None:
        """Equal-weight portfolio should have lower vol than QQQ alone."""
        # Equal weight returns
        portfolio_returns = market_data_df.select(
            ((pl.col("SPY") + pl.col("QQQ") + pl.col("BND")) / 3).alias("portfolio")
        )["portfolio"]

        qqq_vol = market_data_df["QQQ"].std()
        port_vol = portfolio_returns.std()

        assert port_vol < qqq_vol, "Diversified portfolio should have lower vol than QQQ"
