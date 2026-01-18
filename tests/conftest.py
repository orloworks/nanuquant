"""Test fixtures for polars_metrics.

Provides both synthetic deterministic data and real market data (SPY, QQQ, BND).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# Cache directory for downloaded market data
CACHE_DIR = Path(__file__).parent / ".data_cache"


# =============================================================================
# Synthetic Data Fixtures (Deterministic, Fast, No Network)
# =============================================================================


@pytest.fixture(scope="session")
def sample_returns() -> pd.Series:
    """Generate deterministic test data as pandas Series.

    Returns 1000 daily returns with specific patterns injected for edge case testing:
    - Day 0: -5% drawdown (tests first-day drawdown handling)
    - Days 100-104: Win streak [3%, 2%, 1%, 2%, 3%]
    - Days 200-206: Loss streak [-2%] * 7
    - Day 500: +15% outlier win
    - Day 600: -12% outlier loss
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    returns = np.random.normal(0.0005, 0.02, 1000)

    # Inject specific patterns for edge case testing
    returns[0] = -0.05  # First day drawdown
    returns[100:105] = [0.03, 0.02, 0.01, 0.02, 0.03]  # Win streak
    returns[200:207] = [-0.02] * 7  # Loss streak
    returns[500] = 0.15  # Outlier win
    returns[600] = -0.12  # Outlier loss

    return pd.Series(returns, index=dates, name="strategy")


@pytest.fixture(scope="session")
def polars_returns(sample_returns: pd.Series) -> pl.Series:
    """Polars Series version of sample_returns."""
    return pl.Series("returns", sample_returns.values)


@pytest.fixture(scope="session")
def benchmark_returns() -> pd.Series:
    """Synthetic benchmark returns (different seed)."""
    np.random.seed(123)
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    returns = np.random.normal(0.0003, 0.015, 1000)
    return pd.Series(returns, index=dates, name="benchmark")


@pytest.fixture(scope="session")
def polars_benchmark(benchmark_returns: pd.Series) -> pl.Series:
    """Polars Series version of benchmark_returns."""
    return pl.Series("benchmark", benchmark_returns.values)


# =============================================================================
# Real Market Data Fixtures (SPY, QQQ, BND)
# =============================================================================


def _download_and_cache(
    ticker: str,
    start: str = "2019-01-01",
    end: str = "2024-01-01",
) -> pd.Series:
    """Download ticker data and cache to disk.

    Caches data as parquet to avoid repeated API calls and rate limiting.
    """
    import yfinance as yf

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{ticker}_{start}_{end}.parquet"

    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        return df["returns"]

    # Download from Yahoo Finance
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise RuntimeError(f"Failed to download {ticker} data")

    # Handle both old and new yfinance column formats
    if isinstance(data.columns, pd.MultiIndex):
        close = data[("Close", ticker)]
    else:
        close = data["Close"]

    returns = close.pct_change().dropna()
    returns.name = "returns"

    # Cache to disk
    df = pd.DataFrame({"returns": returns})
    df.to_parquet(cache_file)

    return returns


@pytest.fixture(scope="session")
def spy_returns() -> pd.Series:
    """5 years of SPY daily returns (2019-2024).

    S&P 500 ETF - broad US equity market exposure.
    """
    return _download_and_cache("SPY")


@pytest.fixture(scope="session")
def qqq_returns() -> pd.Series:
    """5 years of QQQ daily returns (2019-2024).

    Nasdaq-100 ETF - tech-heavy, higher volatility than SPY.
    """
    return _download_and_cache("QQQ")


@pytest.fixture(scope="session")
def bnd_returns() -> pd.Series:
    """5 years of BND daily returns (2019-2024).

    Vanguard Total Bond Market ETF - low volatility, negative correlation to equities.
    """
    return _download_and_cache("BND")


@pytest.fixture(scope="session")
def spy_polars(spy_returns: pd.Series) -> pl.Series:
    """SPY returns as Polars Series."""
    return pl.Series("SPY", spy_returns.values)


@pytest.fixture(scope="session")
def qqq_polars(qqq_returns: pd.Series) -> pl.Series:
    """QQQ returns as Polars Series."""
    return pl.Series("QQQ", qqq_returns.values)


@pytest.fixture(scope="session")
def bnd_polars(bnd_returns: pd.Series) -> pl.Series:
    """BND returns as Polars Series."""
    return pl.Series("BND", bnd_returns.values)


@pytest.fixture(scope="session")
def market_data_df(
    spy_returns: pd.Series,
    qqq_returns: pd.Series,
    bnd_returns: pd.Series,
) -> pl.DataFrame:
    """DataFrame with all three ETFs aligned by date.

    Useful for portfolio and correlation testing.
    """
    # Align on common dates
    df = pd.DataFrame({
        "SPY": spy_returns,
        "QQQ": qqq_returns,
        "BND": bnd_returns,
    }).dropna()

    return pl.from_pandas(df.reset_index(drop=True))


# =============================================================================
# Edge Case Fixtures
# =============================================================================


@pytest.fixture
def empty_returns() -> pl.Series:
    """Empty returns series for edge case testing."""
    return pl.Series("empty", [], dtype=pl.Float64)


@pytest.fixture
def single_return() -> pl.Series:
    """Single observation for edge case testing."""
    return pl.Series("single", [0.05])


@pytest.fixture
def all_positive_returns() -> pl.Series:
    """All positive returns (no losing days)."""
    return pl.Series("winners", [0.01, 0.02, 0.03, 0.01, 0.02])


@pytest.fixture
def all_negative_returns() -> pl.Series:
    """All negative returns (no winning days)."""
    return pl.Series("losers", [-0.01, -0.02, -0.03, -0.01, -0.02])


@pytest.fixture
def flat_returns() -> pl.Series:
    """All zero returns (flat equity curve)."""
    return pl.Series("flat", [0.0, 0.0, 0.0, 0.0, 0.0])
