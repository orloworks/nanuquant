# NanuQuant

### Institutional quantitative analytics at Polars speed.

[![PyPI](https://img.shields.io/pypi/v/nanuquant)](https://pypi.org/project/nanuquant/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NanuQuant** is a high-performance, native Polars library for quantitative finance. It replaces legacy Pandas-based tools with a vectorized engine capable of handling tick-level data and large-scale backtests without memory overhead.

It goes beyond standard metrics to provide institutional-grade robustness testing, volatility modeling, and execution analysis.

## Why NanuQuant?

- **Zero Pandas Dependency:** Built purely on Polars for maximum speed and stability in production containers.
- **Institutional Rigor:** Includes advanced metrics like Cornish-Fisher VaR, Ledoit-Wolf Shrinkage, and Deflated Sharpe Ratios.
- **Production Ready:** Fully typed, rigorously tested against industry standards (QuantStats), and designed for high-frequency workflows.

## Installation

```bash
pip install nanuquant
```

## Quick Start

```python
import polars as pl
import nanuquant as nq

# 1. Load market data (lazy or eager)
returns = pl.Series("returns", [0.01, -0.02, 0.03, 0.01, -0.01, 0.02])

# 2. Calculate core metrics
sharpe = nq.sharpe(returns, risk_free_rate=0.04)
sortino = nq.sortino(returns)
max_dd = nq.max_drawdown(returns)

print(f"Sharpe: {sharpe:.2f} | Max Drawdown: {max_dd:.2%}")
```

## Institutional Analysis

NanuQuant shines where other libraries stop. Validate your strategies with statistical rigor:

### Robustness Testing

Don't be fooled by random luck. Use the **Probabilistic Sharpe Ratio (PSR)** and **Deflated Sharpe Ratio (DSR)** to adjust for non-normality and multiple testing overfitting.

```python
# Did you test 100 variations of your strategy? Adjust for selection bias.
dsr = nq.deflated_sharpe_ratio(returns, n_trials=100)
print(f"Deflated Sharpe Probability: {dsr:.4f}")
```

### Volatility Modeling

Detect volatility clustering and regime changes using GARCH(1,1) estimates.

```python
from nanuquant import institutional

# Test for ARCH effects (volatility clustering)
arch_test = institutional.arch_effect_test(returns)
if arch_test.has_arch_effects:
    print("Regime: Volatility Clustering Detected")
```

## Features

### Core Metrics

* **Returns:** CAGR, Compounded Growth, Log Returns, Greeks (Alpha/Beta)
* **Risk:** Volatility, VaR (Parametric, Historical, Cornish-Fisher), CVaR, Ulcer Index
* **Performance:** Sharpe, Sortino, Calmar, Omega, Kelly Criterion
* **Drawdowns:** Max Drawdown, Drawdown Duration, Underwater Curves

### Advanced Trading

* **System Quality:** SQN, Expectancy, K-Ratio
* **Trade Analysis:** Win Rate, Profit Factor, Payoff Ratio, Exposure

### Institutional & Systemic

* **Execution:** Implementation Shortfall, Market Impact (Square-root law)
* **Portfolio:** Ledoit-Wolf Covariance, Marginal Contribution to Risk (MCR)
* **Systemic:** Absorption Ratio, Tail Dependence

## DataFrame Namespace

Import `nanuquant` to register the `.metrics` namespace on Polars expressions:

```python
import polars as pl
import nanuquant as nq  # Registers the .metrics accessor

df = pl.DataFrame({
    "returns": [0.01, -0.02, 0.015, -0.01, 0.02] * 50
})

# Single metric
result = df.select(pl.col("returns").metrics.sharpe())

# Multiple metrics in one select
metrics_df = df.select([
    pl.col("returns").metrics.sharpe().alias("sharpe"),
    pl.col("returns").metrics.sortino().alias("sortino"),
    pl.col("returns").metrics.max_drawdown().alias("max_dd"),
    pl.col("returns").metrics.volatility().alias("volatility"),
])

# Rolling metrics with with_columns
df_with_rolling = df.with_columns([
    pl.col("returns").metrics.rolling_volatility().alias("rolling_vol"),
    pl.col("returns").metrics.rolling_sharpe().alias("rolling_sharpe"),
])
```

## Null Handling

NanuQuant follows QuantStats/pandas conventions by dropping null values before calculations by default. This ensures consistent results across different data sources:

```python
import polars as pl
import nanuquant as nq

# Nulls are dropped by default
returns_with_nulls = pl.Series([0.01, None, -0.02, None, 0.015])
sharpe = nq.sharpe(returns_with_nulls)  # Calculates using [0.01, -0.02, 0.015]
```

## Available Metrics

### Returns (11)
`comp`, `cagr`, `avg_return`, `avg_win`, `avg_loss`, `best`, `worst`, `win_rate`, `payoff_ratio`, `profit_factor`, `consecutive_wins`, `consecutive_losses`

### Risk (7)
`volatility`, `var`, `cvar`, `max_drawdown`, `to_drawdown_series`, `ulcer_index`, `downside_deviation`

### Performance (14)
`sharpe`, `sortino`, `calmar`, `omega`, `gain_to_pain_ratio`, `ulcer_performance_index`, `kelly_criterion`, `tail_ratio`, `common_sense_ratio`, `risk_return_ratio`, `recovery_factor`, `greeks`, `information_ratio`, `r_squared`, `treynor_ratio`, `benchmark_correlation`

### Distribution (8)
`skewness`, `kurtosis`, `jarque_bera`, `shapiro_wilk`, `outlier_win_ratio`, `outlier_loss_ratio`, `expected_return`, `geometric_mean`

### Rolling (5)
`rolling_volatility`, `rolling_sharpe`, `rolling_sortino`, `rolling_beta`, `rolling_greeks`

### Trading (12)
`exposure`, `ghpr`, `rar`, `cpc_index`, `serenity_index`, `risk_of_ruin`, `adjusted_sortino`, `smart_sharpe`, `smart_sortino`, `sqn`, `expectancy`, `k_ratio`

### Institutional (2)
`deflated_sharpe_ratio`, `probabilistic_sharpe_ratio`

## Known Differences from QuantStats

This library intentionally differs from QuantStats in some areas for improved consistency:

1. **CAGR / Calmar / Treynor**: Uses periods-based calculation instead of datetime index, working with any time series (not just datetime-indexed).

2. **Omega Ratio**: Fixes a bug in some QuantStats versions where `Series.sum().values[0]` fails.

3. **CPC Index**: Uses the standard formula from trading literature: `profit_factor * win_rate * payoff_ratio`.

4. **Smart Sharpe/Sortino**: Uses Lo (2002) autocorrelation adjustment formula.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run differential tests against QuantStats
pytest tests/test_vs_quantstats.py -v

# Run integration tests with real market data
pytest -m integration

# Type checking
mypy nanuquant
```

## License

MIT License. See [LICENSE](LICENSE) for details.
