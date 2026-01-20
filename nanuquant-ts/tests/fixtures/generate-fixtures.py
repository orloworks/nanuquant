#!/usr/bin/env python3
"""Generate test fixtures for NanuQuant TypeScript tests.

This script generates deterministic test data using the Python NanuQuant library
and saves the inputs and expected outputs as JSON files for TypeScript tests.

Usage:
    python generate-fixtures.py
"""

import json
import sys
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import polars as pl

# Add parent directory to path for nanuquant import
sys.path.insert(0, str(Path(__file__).parents[3]))

from nanuquant.core import returns, risk, performance, rolling
from nanuquant.core.distribution import skewness, kurtosis, expected_return, geometric_mean, outlier_win_ratio, outlier_loss_ratio
from nanuquant.core.timeseries import cumulative_returns, histogram
from nanuquant.core.periods import monthly_returns


def generate_returns(seed: int = 42, n: int = 1000) -> np.ndarray:
    """Generate deterministic random returns."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.02, n)


def generate_benchmark(seed: int = 123, n: int = 1000) -> np.ndarray:
    """Generate deterministic benchmark returns."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0003, 0.015, n)


def generate_dates(n: int = 1000) -> list[str]:
    """Generate a sequence of dates."""
    start = date(2020, 1, 1)
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def to_json_safe(value):
    """Convert value to JSON-safe format."""
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [to_json_safe(v) for v in value]
    if isinstance(value, pl.Series):
        return [to_json_safe(v) for v in value.to_list()]
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def main():
    output_dir = Path(__file__).parent / "expected"
    output_dir.mkdir(exist_ok=True)

    # Generate test data
    returns_data = generate_returns()
    benchmark_data = generate_benchmark()
    dates = generate_dates()

    # Save input data
    inputs = {
        "returns": returns_data.tolist(),
        "benchmark": benchmark_data.tolist(),
        "dates": dates,
        "seed": 42,
        "n": 1000,
    }

    with open(Path(__file__).parent / "inputs.json", "w") as f:
        json.dump(inputs, f, indent=2)

    # Convert to Polars for Python functions
    returns_series = pl.Series(returns_data)
    benchmark_series = pl.Series(benchmark_data)
    dates_series = pl.Series([date.fromisoformat(d) for d in dates])

    # Generate returns metrics
    returns_expected = {
        "comp": to_json_safe(returns.comp(returns_series)),
        "cagr": to_json_safe(returns.cagr(returns_series, periods_per_year=252)),
        "avgReturn": to_json_safe(returns.avg_return(returns_series)),
        "avgWin": to_json_safe(returns.avg_win(returns_series)),
        "avgLoss": to_json_safe(returns.avg_loss(returns_series)),
        "best": to_json_safe(returns.best(returns_series)),
        "worst": to_json_safe(returns.worst(returns_series)),
        "winRate": to_json_safe(returns.win_rate(returns_series)),
        "payoffRatio": to_json_safe(returns.payoff_ratio(returns_series)),
        "profitFactor": to_json_safe(returns.profit_factor(returns_series)),
        "consecutiveWins": to_json_safe(returns.consecutive_wins(returns_series)),
        "consecutiveLosses": to_json_safe(returns.consecutive_losses(returns_series)),
    }

    with open(output_dir / "returns.json", "w") as f:
        json.dump(returns_expected, f, indent=2)

    # Generate risk metrics
    risk_expected = {
        "volatility": to_json_safe(risk.volatility(returns_series, periods_per_year=252)),
        "volatilityNotAnnualized": to_json_safe(risk.volatility(returns_series, annualize=False)),
        "var95": to_json_safe(risk.var(returns_series, confidence=0.95)),
        "var99": to_json_safe(risk.var(returns_series, confidence=0.99)),
        "cvar95": to_json_safe(risk.cvar(returns_series, confidence=0.95)),
        "cvar99": to_json_safe(risk.cvar(returns_series, confidence=0.99)),
        "maxDrawdown": to_json_safe(risk.max_drawdown(returns_series)),
        "ulcerIndex": to_json_safe(risk.ulcer_index(returns_series)),
        "downsideDeviation": to_json_safe(risk.downside_deviation(returns_series, periods_per_year=252)),
        "drawdownSeries": to_json_safe(risk.to_drawdown_series(returns_series)),
    }

    with open(output_dir / "risk.json", "w") as f:
        json.dump(risk_expected, f, indent=2)

    # Generate performance metrics
    performance_expected = {
        "sharpe": to_json_safe(performance.sharpe(returns_series, periods_per_year=252)),
        "sharpeRf": to_json_safe(performance.sharpe(returns_series, risk_free_rate=0.02, periods_per_year=252)),
        "sortino": to_json_safe(performance.sortino(returns_series, periods_per_year=252)),
        "calmar": to_json_safe(performance.calmar(returns_series, periods_per_year=252)),
        "omega": to_json_safe(performance.omega(returns_series)),
        "gainToPainRatio": to_json_safe(performance.gain_to_pain_ratio(returns_series)),
        "ulcerPerformanceIndex": to_json_safe(performance.ulcer_performance_index(returns_series)),
        "kellyCriterion": to_json_safe(performance.kelly_criterion(returns_series)),
        "tailRatio": to_json_safe(performance.tail_ratio(returns_series)),
        "commonSenseRatio": to_json_safe(performance.common_sense_ratio(returns_series)),
        "riskReturnRatio": to_json_safe(performance.risk_return_ratio(returns_series)),
        "recoveryFactor": to_json_safe(performance.recovery_factor(returns_series)),
        "greeks": {
            "alpha": to_json_safe(performance.greeks(returns_series, benchmark_series)[0]),
            "beta": to_json_safe(performance.greeks(returns_series, benchmark_series)[1]),
        },
        "informationRatio": to_json_safe(performance.information_ratio(returns_series, benchmark_series)),
        "rSquared": to_json_safe(performance.r_squared(returns_series, benchmark_series)),
        "treynorRatio": to_json_safe(performance.treynor_ratio(returns_series, benchmark_series, periods_per_year=252)),
        "benchmarkCorrelation": to_json_safe(performance.benchmark_correlation(returns_series, benchmark_series)),
    }

    with open(output_dir / "performance.json", "w") as f:
        json.dump(performance_expected, f, indent=2)

    # Generate correlation metrics (using numpy for correlation/covariance)
    correlation_expected = {
        "correlation": to_json_safe(np.corrcoef(returns_data, benchmark_data)[0, 1]),
        "covariance": to_json_safe(np.cov(returns_data, benchmark_data, ddof=1)[0, 1]),
    }

    with open(output_dir / "correlation.json", "w") as f:
        json.dump(correlation_expected, f, indent=2)

    # Generate distribution metrics
    distribution_expected = {
        "skewness": to_json_safe(skewness(returns_series)),
        "kurtosis": to_json_safe(kurtosis(returns_series)),
        "expectedReturn": to_json_safe(expected_return(returns_series)),
        "geometricMean": to_json_safe(geometric_mean(returns_series)),
        "outlierWinRatio": to_json_safe(outlier_win_ratio(returns_series)),
        "outlierLossRatio": to_json_safe(outlier_loss_ratio(returns_series)),
    }

    with open(output_dir / "distribution.json", "w") as f:
        json.dump(distribution_expected, f, indent=2)

    # Generate rolling metrics (take last 100 values to keep fixture size reasonable)
    rolling_expected = {
        "rollingVolatility": to_json_safe(rolling.rolling_volatility(returns_series, rolling_period=126)[-100:]),
        "rollingSharpe": to_json_safe(rolling.rolling_sharpe(returns_series, rolling_period=126)[-100:]),
        "rollingSortino": to_json_safe(rolling.rolling_sortino(returns_series, rolling_period=126)[-100:]),
        "rollingBeta": to_json_safe(rolling.rolling_beta(returns_series, benchmark_series, rolling_period=126)[-100:]),
    }

    greeks_result = rolling.rolling_greeks(returns_series, benchmark_series, rolling_period=126)
    rolling_expected["rollingAlpha"] = to_json_safe(greeks_result["rolling_alpha"][-100:])
    rolling_expected["rollingBetaGreeks"] = to_json_safe(greeks_result["rolling_beta"][-100:])

    with open(output_dir / "rolling.json", "w") as f:
        json.dump(rolling_expected, f, indent=2)

    # Generate timeseries metrics
    timeseries_expected = {
        "cumulativeReturns": to_json_safe(cumulative_returns(returns_series)[-100:]),
    }

    # Histogram (use fewer bins for fixture)
    hist_df = histogram(returns_series, bins=20)
    timeseries_expected["histogram"] = {
        "binStart": hist_df["bin_start"].to_list(),
        "binEnd": hist_df["bin_end"].to_list(),
        "binCenter": hist_df["bin_center"].to_list(),
        "count": hist_df["count"].to_list(),
        "frequency": hist_df["frequency"].to_list(),
    }

    with open(output_dir / "timeseries.json", "w") as f:
        json.dump(timeseries_expected, f, indent=2)

    print("✓ Generated test fixtures in", output_dir)
    print(f"  - inputs.json ({len(returns_data)} data points)")
    print(f"  - expected/returns.json ({len(returns_expected)} metrics)")
    print(f"  - expected/risk.json ({len(risk_expected)} metrics)")
    print(f"  - expected/performance.json ({len(performance_expected)} metrics)")
    print(f"  - expected/correlation.json ({len(correlation_expected)} metrics)")
    print(f"  - expected/distribution.json ({len(distribution_expected)} metrics)")
    print(f"  - expected/rolling.json ({len(rolling_expected)} metrics)")
    print(f"  - expected/timeseries.json ({len(timeseries_expected)} metrics)")


if __name__ == "__main__":
    main()
