# Quick Start Guide

> **Important**: NanuQuant is for educational and research purposes only. The metrics and analyses provided do **NOT** constitute financial advice. Always consult a qualified financial professional before making investment decisions. See [DISCLAIMER](https://github.com/launchstack-dev/nanuquant/blob/main/DISCLAIMER.md).

## Introduction

NanuQuant is a high-performance quantitative finance library built natively on Polars. This guide will walk you through the basic usage patterns.

## Basic Usage

### Import and Create Data

```python
import polars as pl
import nanuquant as nq

# Create a simple return series
# These are period-over-period returns (e.g., daily returns)
returns = pl.Series("returns", [0.01, -0.02, 0.03, 0.01, -0.01, 0.02, -0.005, 0.015])
```

### Calculate Individual Metrics

```python
# Core performance metrics
sharpe = nq.sharpe(returns)
sortino = nq.sortino(returns)
max_dd = nq.max_drawdown(returns)

print(f"Sharpe Ratio: {sharpe:.4f}")
print(f"Sortino Ratio: {sortino:.4f}")
print(f"Max Drawdown: {max_dd:.2%}")
```

### Calculate Multiple Metrics at Once

```python
# Using the full_metrics function for comprehensive analysis
from nanuquant.reports import full_metrics

report = full_metrics(returns)

# Access categorized metrics
print(f"CAGR: {report.returns['cagr']:.2%}")
print(f"Volatility: {report.risk['volatility']:.2%}")
print(f"Sharpe: {report.performance['sharpe']:.4f}")
print(f"Skewness: {report.distribution['skewness']:.4f}")
```

## Working with DataFrames

### Using the .metrics Namespace

NanuQuant registers a `.metrics` namespace on Polars expressions:

```python
import polars as pl
import nanuquant as nq  # Importing registers the namespace

# Create DataFrame with returns
df = pl.DataFrame({
    "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 3, 31), eager=True),
    "returns": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02, -0.005, 0.015] * 11 + [0.01, 0.02, 0.01]
})

# Calculate single metric
result = df.select(pl.col("returns").metrics.sharpe())
print(result)

# Calculate multiple metrics in one pass
metrics_df = df.select([
    pl.col("returns").metrics.sharpe().alias("sharpe"),
    pl.col("returns").metrics.sortino().alias("sortino"),
    pl.col("returns").metrics.max_drawdown().alias("max_drawdown"),
    pl.col("returns").metrics.volatility().alias("volatility"),
])
print(metrics_df)
```

### Rolling Metrics

Calculate metrics over a rolling window:

```python
# Add rolling metrics as new columns
df_rolling = df.with_columns([
    pl.col("returns").metrics.rolling_sharpe(window=20).alias("rolling_sharpe"),
    pl.col("returns").metrics.rolling_volatility(window=20).alias("rolling_vol"),
])

print(df_rolling.tail())
```

## Benchmark Comparison

Compare strategy returns against a benchmark:

```python
# Strategy and benchmark returns
strategy = pl.Series("strategy", [0.02, -0.01, 0.03, 0.01, -0.02, 0.015])
benchmark = pl.Series("benchmark", [0.01, -0.005, 0.02, 0.005, -0.01, 0.01])

# Calculate benchmark-relative metrics
alpha, beta = nq.greeks(strategy, benchmark)
info_ratio = nq.information_ratio(strategy, benchmark)
r_squared = nq.r_squared(strategy, benchmark)

print(f"Alpha: {alpha:.4f}")
print(f"Beta: {beta:.4f}")
print(f"Information Ratio: {info_ratio:.4f}")
print(f"R-squared: {r_squared:.4f}")
```

## Risk Analysis

### Value at Risk (VaR)

```python
# Different VaR methodologies
var_95 = nq.var(returns, confidence=0.95)  # Parametric VaR
cvar_95 = nq.cvar(returns, confidence=0.95)  # Conditional VaR (Expected Shortfall)

print(f"95% VaR: {var_95:.2%}")
print(f"95% CVaR: {cvar_95:.2%}")

# Advanced VaR (institutional module)
from nanuquant import institutional

cf_var = institutional.cornish_fisher_var(returns, confidence=0.95)
print(f"95% Cornish-Fisher VaR: {cf_var:.2%}")
```

### Drawdown Analysis

```python
# Maximum drawdown
max_dd = nq.max_drawdown(returns)
print(f"Maximum Drawdown: {max_dd:.2%}")

# Detailed drawdown information
from nanuquant.core.timeseries import drawdown_details

dd_info = drawdown_details(returns, top_n=3)
print(dd_info)
```

## Configuration

### Global Configuration

```python
from nanuquant.config import get_config, set_config, MetricsConfig

# View current configuration
config = get_config()
print(f"Risk-free rate: {config.risk_free_rate}")
print(f"Periods per year: {config.periods_per_year}")

# Set custom configuration
custom_config = MetricsConfig(
    risk_free_rate=0.05,  # 5% risk-free rate
    periods_per_year=252,  # Daily data
)
set_config(custom_config)

# Now all metrics use the new config
sharpe = nq.sharpe(returns)  # Uses 5% risk-free rate
```

### Per-Call Configuration

Override configuration for specific calls:

```python
# Most functions accept parameters directly
sharpe_rf0 = nq.sharpe(returns, risk_free_rate=0.0)
sharpe_rf5 = nq.sharpe(returns, risk_free_rate=0.05)

print(f"Sharpe (RF=0%): {sharpe_rf0:.4f}")
print(f"Sharpe (RF=5%): {sharpe_rf5:.4f}")
```

## Handling Different Frequencies

NanuQuant supports various data frequencies:

```python
from nanuquant.config import MetricsConfig, set_config

# For daily data (default)
daily_config = MetricsConfig(periods_per_year=252, frequency="D")

# For weekly data
weekly_config = MetricsConfig(periods_per_year=52, frequency="W")

# For monthly data
monthly_config = MetricsConfig(periods_per_year=12, frequency="M")

# For hourly data
hourly_config = MetricsConfig(periods_per_year=252 * 6.5, frequency="H")  # 6.5 trading hours/day
```

## Generating Reports

### Compute All Metrics

```python
from nanuquant.reports import full_metrics

# Define your returns and an optional benchmark
returns = pl.Series("returns", [0.01, -0.02, 0.03, 0.01, -0.01, 0.02, -0.005, 0.015])
benchmark = pl.Series("benchmark", [0.008, -0.015, 0.02, 0.008, -0.008, 0.015, -0.004, 0.01])

# Get comprehensive metrics report
report = full_metrics(
    returns,
    benchmark=benchmark,  # Optional benchmark
)

# Access by category
print("Returns Metrics:", report.returns)
print("Risk Metrics:", report.risk)
print("Performance Metrics:", report.performance)
print("Distribution Metrics:", report.distribution)
print("Trading Metrics:", report.trading)
print("Benchmark Metrics:", report.benchmark)
```

### Generate HTML Report

```python
from nanuquant.reports import generate_html_report, save_html_report

# Generate HTML content (using returns and benchmark from above)
html = generate_html_report(returns, benchmark=benchmark, title="My Strategy Analysis")

# Save to file
save_html_report(returns, "strategy_report.html", benchmark=benchmark)
```

## Null Handling

NanuQuant automatically handles null values by dropping them (consistent with QuantStats):

```python
# Nulls are automatically dropped
returns_with_nulls = pl.Series([0.01, None, -0.02, None, 0.015, 0.02])

# This works seamlessly
sharpe = nq.sharpe(returns_with_nulls)
print(f"Sharpe (nulls dropped): {sharpe:.4f}")
```

## Converting Prices to Returns

If you have price data instead of returns:

```python
from nanuquant.core.utils import simple_returns, log_returns

prices = pl.Series("price", [100, 102, 99, 103, 101, 105])

# Simple percentage returns
simple_rets = simple_returns(prices)
print("Simple returns:", simple_rets)

# Log returns (continuously compounded)
log_rets = log_returns(prices)
print("Log returns:", log_rets)
```

## Error Handling

NanuQuant raises specific exceptions for invalid inputs:

```python
from nanuquant.exceptions import (
    EmptySeriesError,
    InsufficientDataError,
    BenchmarkMismatchError,
)

# Empty series
try:
    nq.sharpe(pl.Series([]))
except EmptySeriesError as e:
    print(f"Error: {e}")

# Insufficient data
try:
    nq.rolling_sharpe(pl.Series([0.01, 0.02]), window=100)
except InsufficientDataError as e:
    print(f"Error: {e}")
```

## Next Steps

- [API Reference](api/core.md) - Complete function documentation
- [Mathematics](mathematics.md) - Understanding the formulas
- [Use Cases](use-cases.md) - Real-world applications
- [Testing](testing.md) - How NanuQuant is validated

---

> **Remember**: All metrics and analyses are for educational and research purposes only. Past performance does not guarantee future results. Always verify calculations independently and consult with qualified professionals before making financial decisions.
