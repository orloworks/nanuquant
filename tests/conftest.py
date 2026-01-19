"""Test fixtures for nanuquant.

Provides both synthetic deterministic data and real market data (SPY, QQQ, BND).
Market data is pre-cached in tests/.data_cache/ - no network required.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

if TYPE_CHECKING:
    pass

# Cache directory for market data (committed to repo)
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
# Real Market Data Loading (Pre-cached, No Network Required)
# =============================================================================


def load_cached_returns(ticker: str) -> pd.Series:
    """Load pre-cached returns from parquet file.

    Data is committed to repo in tests/.data_cache/
    Available tickers: SPY, QQQ, BND
    """
    cache_file = CACHE_DIR / f"{ticker}_max.parquet"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Cached data not found: {cache_file}\n"
            f"Available files: {list(CACHE_DIR.glob('*.parquet'))}"
        )
    df = pd.read_parquet(cache_file)
    return df["returns"]


def slice_returns(
    returns: pd.Series,
    start: str | None = None,
    end: str | None = None,
    years: int | None = None,
) -> pd.Series:
    """Slice returns by date range or recent years.

    Parameters
    ----------
    returns : pd.Series
        Full returns series with DatetimeIndex.
    start : str, optional
        Start date (e.g., "2020-01-01").
    end : str, optional
        End date (e.g., "2023-12-31").
    years : int, optional
        Use most recent N years. Overrides start/end.

    Returns
    -------
    pd.Series
        Sliced returns.

    Examples
    --------
    >>> spy = load_cached_returns("SPY")
    >>> slice_returns(spy, start="2020-01-01", end="2022-12-31")  # 3 years
    >>> slice_returns(spy, years=5)  # Last 5 years
    """
    if years is not None:
        end_date = returns.index[-1]
        start_date = end_date - pd.DateOffset(years=years)
        return returns[start_date:end_date]

    if start is not None or end is not None:
        return returns[start:end]

    return returns


# =============================================================================
# Real Market Data Fixtures (SPY, QQQ, BND)
# =============================================================================


@pytest.fixture(scope="session")
def spy_returns_full() -> pd.Series:
    """Full SPY history (1993-present, ~8000+ days).

    S&P 500 ETF - broad US equity market exposure.
    """
    return load_cached_returns("SPY")


@pytest.fixture(scope="session")
def qqq_returns_full() -> pd.Series:
    """Full QQQ history (1999-present, ~6700+ days).

    Nasdaq-100 ETF - tech-heavy, higher volatility than SPY.
    """
    return load_cached_returns("QQQ")


@pytest.fixture(scope="session")
def bnd_returns_full() -> pd.Series:
    """Full BND history (2007-present, ~4700+ days).

    Vanguard Total Bond Market ETF - low volatility, bonds.
    """
    return load_cached_returns("BND")


# Convenience fixtures for common time windows
@pytest.fixture(scope="session")
def spy_returns(spy_returns_full: pd.Series) -> pd.Series:
    """SPY returns - last 5 years."""
    return slice_returns(spy_returns_full, years=5)


@pytest.fixture(scope="session")
def qqq_returns(qqq_returns_full: pd.Series) -> pd.Series:
    """QQQ returns - last 5 years."""
    return slice_returns(qqq_returns_full, years=5)


@pytest.fixture(scope="session")
def bnd_returns(bnd_returns_full: pd.Series) -> pd.Series:
    """BND returns - last 5 years."""
    return slice_returns(bnd_returns_full, years=5)


# Polars versions
@pytest.fixture(scope="session")
def spy_polars(spy_returns: pd.Series) -> pl.Series:
    """SPY returns as Polars Series (5 years)."""
    return pl.Series("SPY", spy_returns.values)


@pytest.fixture(scope="session")
def qqq_polars(qqq_returns: pd.Series) -> pl.Series:
    """QQQ returns as Polars Series (5 years)."""
    return pl.Series("QQQ", qqq_returns.values)


@pytest.fixture(scope="session")
def bnd_polars(bnd_returns: pd.Series) -> pl.Series:
    """BND returns as Polars Series (5 years)."""
    return pl.Series("BND", bnd_returns.values)


# Full history Polars versions
@pytest.fixture(scope="session")
def spy_polars_full(spy_returns_full: pd.Series) -> pl.Series:
    """SPY full history as Polars Series."""
    return pl.Series("SPY", spy_returns_full.values)


@pytest.fixture(scope="session")
def qqq_polars_full(qqq_returns_full: pd.Series) -> pl.Series:
    """QQQ full history as Polars Series."""
    return pl.Series("QQQ", qqq_returns_full.values)


@pytest.fixture(scope="session")
def bnd_polars_full(bnd_returns_full: pd.Series) -> pl.Series:
    """BND full history as Polars Series."""
    return pl.Series("BND", bnd_returns_full.values)


@pytest.fixture(scope="session")
def market_data_df(
    spy_returns: pd.Series,
    qqq_returns: pd.Series,
    bnd_returns: pd.Series,
) -> pl.DataFrame:
    """DataFrame with all three ETFs aligned by date (5 years).

    Useful for portfolio and correlation testing.
    """
    # Align on common dates
    df = pd.DataFrame({
        "SPY": spy_returns,
        "QQQ": qqq_returns,
        "BND": bnd_returns,
    }).dropna()

    return pl.from_pandas(df.reset_index(drop=True))


@pytest.fixture(scope="session")
def market_data_df_full(
    spy_returns_full: pd.Series,
    qqq_returns_full: pd.Series,
    bnd_returns_full: pd.Series,
) -> pl.DataFrame:
    """DataFrame with all three ETFs aligned - full common history.

    Common period starts from BND inception (2007).
    """
    df = pd.DataFrame({
        "SPY": spy_returns_full,
        "QQQ": qqq_returns_full,
        "BND": bnd_returns_full,
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


# =============================================================================
# Time Window Parametrization Helpers
# =============================================================================


def get_time_windows() -> list[tuple[str, int]]:
    """Return list of (name, years) tuples for parametrized tests."""
    return [
        ("1Y", 1),
        ("3Y", 3),
        ("5Y", 5),
        ("10Y", 10),
        ("20Y", 20),
    ]
