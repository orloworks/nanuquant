# Polars Metrics

Native Polars metrics library for quantitative finance - a complete QuantStats replacement.

## Features

- **67 QuantStats metrics** fully replicated with native Polars
- **18 additional trading metrics** (SQN, Expectancy, K-ratio, etc.)
- **Institutional metrics** (Deflated Sharpe, Probabilistic Sharpe)
- **Zero pandas dependency** in production code path
- **Lazy evaluation support** for large datasets
- **Polars expression namespace** for idiomatic usage
- **Full type annotations** with py.typed marker for MyPy support

## Installation

```bash
pip install polars-metrics
```

## Quick Start

### Functional API

```python
import polars as pl
import polars_metrics as pm

# Generate sample returns
returns = pl.Series("returns", [0.01, -0.02, 0.03, 0.01, -0.01, 0.02])

# Core metrics
sharpe = pm.sharpe(returns, risk_free_rate=0.02, periods_per_year=252)
sortino = pm.sortino(returns, risk_free_rate=0.0, periods_per_year=252)
max_dd = pm.max_drawdown(returns)
vol = pm.volatility(returns, periods_per_year=252)

# Benchmark metrics (with benchmark series)
benchmark = pl.Series("benchmark", [0.005, -0.01, 0.02, 0.005, -0.005, 0.01])
alpha, beta = pm.greeks(returns, benchmark, periods_per_year=252)
info_ratio = pm.information_ratio(returns, benchmark)

# Rolling metrics
rolling_vol = pm.rolling_volatility(returns, rolling_period=126)
rolling_sharpe = pm.rolling_sharpe(returns, rolling_period=126)
```

### DataFrame Namespace

Import `polars_metrics` to register the `.metrics` namespace on Polars expressions:

```python
import polars as pl
import polars_metrics as pm  # Registers the .metrics accessor

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

### Null Handling

Polars Metrics follows QuantStats/pandas conventions by dropping null values before calculations by default. This ensures consistent results across different data sources:

```python
import polars as pl
import polars_metrics as pm

# Nulls are dropped by default
returns_with_nulls = pl.Series([0.01, None, -0.02, None, 0.015])
sharpe = pm.sharpe(returns_with_nulls)  # Calculates using [0.01, -0.02, 0.015]
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

### Rolling (4)
`rolling_volatility`, `rolling_sharpe`, `rolling_sortino`, `rolling_beta`

### Trading (11)
`exposure`, `ghpr`, `rar`, `cpc_index`, `serenity_index`, `risk_of_ruin`, `adjusted_sortino`, `smart_sharpe`, `smart_sortino`, `sqn`, `expectancy`, `k_ratio`

### Institutional (2)
`deflated_sharpe_ratio`, `probabilistic_sharpe_ratio`

## Known Differences from QuantStats

This library intentionally differs from QuantStats in some areas for improved consistency:

1. **CAGR / Calmar / Treynor**: Uses periods-based calculation instead of datetime index, working with any time series (not just datetime-indexed).

2. **Omega Ratio**: Fixes a bug in some QuantStats versions where `Series.sum().values[0]` fails.

3. **CPC Index**: Uses the standard formula from trading literature: `profit_factor * win_rate * payoff_ratio`.

4. **Smart Sharpe/Sortino**: Uses Lo (2002) autocorrelation adjustment formula.

See the test documentation in `tests/test_vs_quantstats.py` for detailed comparison notes.

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
mypy polars_metrics
```

## License

MIT
