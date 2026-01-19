# NanuQuant

### Institutional quantitative analytics at Polars speed.

[![PyPI](https://img.shields.io/pypi/v/nanuquant)](https://pypi.org/project/nanuquant/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

> **DISCLAIMER**: NanuQuant is for **educational and research purposes only**. Nothing in this library constitutes financial advice. Past performance does not guarantee future results. See [DISCLAIMER.md](DISCLAIMER.md) for full legal notices.

---

## About the Name

In Inuktitut, the language of the Inuit people, **"nanuq"** means **polar bear**. Since NanuQuant is built entirely on [Polars](https://pola.rs/) for maximum speed and efficiency, we chose this name to honor the Inuit people while celebrating our foundation on the Polars ecosystem.

---

## Overview

**NanuQuant** is a high-performance, native Polars library for quantitative finance. It replaces legacy Pandas-based tools with a vectorized engine capable of handling tick-level data and large-scale backtests without memory overhead.

It goes beyond standard metrics to provide institutional-grade robustness testing, volatility modeling, and execution analysis.

### Why NanuQuant?

- **Zero Pandas Dependency:** Built purely on Polars for maximum speed and stability in production containers
- **Institutional Rigor:** Includes advanced metrics like Cornish-Fisher VaR, Ledoit-Wolf Shrinkage, and Deflated Sharpe Ratios
- **Production Ready:** Fully typed, rigorously tested against industry standards (QuantStats), and designed for high-frequency workflows
- **Educational Focus:** Comprehensive documentation explaining the mathematics behind every metric

---

## Documentation

| Document | Description |
|----------|-------------|
| [Installation Guide](docs/installation.md) | Setup and configuration |
| [Quick Start](docs/quickstart.md) | Get started in minutes |
| [Core API Reference](docs/api/core.md) | Returns, risk, performance metrics |
| [Advanced API Reference](docs/api/advanced.md) | Trading system metrics |
| [Institutional API Reference](docs/api/institutional.md) | PSR, DSR, GARCH, VaR extensions |
| [Mathematical Foundations](docs/mathematics.md) | Formulas and theory |
| [Testing Methodology](docs/testing.md) | How NanuQuant is validated |
| [Use Cases](docs/use-cases.md) | Practical examples with caveats |
| [Full Disclaimer](DISCLAIMER.md) | Important legal notices |

---

## Installation

```bash
pip install nanuquant
```

For additional features:

```bash
# With HTML report generation
pip install nanuquant[reports]

# For development/testing
pip install nanuquant[dev]

# Everything
pip install nanuquant[all]
```

---

## Quick Start

```python
import polars as pl
import nanuquant as nq

# Load or create return data
returns = pl.Series("returns", [0.01, -0.02, 0.03, 0.01, -0.01, 0.02])

# Calculate core metrics
sharpe = nq.sharpe(returns, risk_free_rate=0.04)
sortino = nq.sortino(returns)
max_dd = nq.max_drawdown(returns)

print(f"Sharpe: {sharpe:.2f}")
print(f"Sortino: {sortino:.2f}")
print(f"Max Drawdown: {max_dd:.2%}")
```

> **Note**: These metrics are for educational analysis only. See [Use Cases](docs/use-cases.md) for proper interpretation.

---

## Features at a Glance

### Core Metrics (60+)

| Category | Metrics |
|----------|---------|
| **Returns** | CAGR, total return, avg win/loss, win rate, profit factor, payoff ratio |
| **Risk** | Volatility, VaR, CVaR, max drawdown, Ulcer Index, downside deviation |
| **Performance** | Sharpe, Sortino, Calmar, Omega, Information Ratio, Treynor, Alpha/Beta |
| **Distribution** | Skewness, kurtosis, Jarque-Bera test, outlier detection |
| **Rolling** | Rolling Sharpe, rolling volatility, rolling beta |

### Advanced Trading Metrics

- **System Quality Number (SQN)** - Trading system quality assessment
- **Expectancy** - Expected value per trade
- **K-Ratio** - Equity curve consistency
- **Smart Sharpe/Sortino** - Autocorrelation-adjusted ratios
- **Risk of Ruin** - Account depletion probability

### Institutional-Grade Analytics

- **Probabilistic Sharpe Ratio (PSR)** - Statistical significance of Sharpe
- **Deflated Sharpe Ratio (DSR)** - Multiple testing adjustment
- **GARCH Volatility** - Conditional volatility modeling
- **Cornish-Fisher VaR** - Skewness/kurtosis-adjusted VaR
- **Ledoit-Wolf Covariance** - Shrinkage estimator for portfolios
- **Absorption Ratio** - Systemic risk measurement

---

## Institutional Analysis Examples

### Robustness Testing

Don't be fooled by random luck. Validate your strategies with statistical rigor:

```python
from nanuquant import institutional

# Did you test 100 variations of your strategy? Adjust for selection bias.
dsr = institutional.deflated_sharpe_ratio(returns, n_trials=100)
print(f"Deflated Sharpe Probability: {dsr:.4f}")

# Check if Sharpe is statistically significant
psr = institutional.probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
print(f"Probability SR > 0: {psr.psr:.2%}")
```

> **Warning**: Even statistically significant backtests can fail in live trading due to regime changes and overfitting.

### Volatility Modeling

Detect volatility clustering and regime changes:

```python
from nanuquant import institutional

# Test for ARCH effects (volatility clustering)
arch_test = institutional.arch_effect_test(returns)
if arch_test.has_arch_effects:
    print("Volatility Clustering Detected")

    # Fit GARCH model
    garch = institutional.garch_volatility(returns)
    print(f"Persistence: {garch.persistence:.4f}")
    print(f"Current Vol Forecast: {garch.forecast:.2%}")
```

---

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

# Multiple metrics in one pass
metrics_df = df.select([
    pl.col("returns").metrics.sharpe().alias("sharpe"),
    pl.col("returns").metrics.sortino().alias("sortino"),
    pl.col("returns").metrics.max_drawdown().alias("max_dd"),
])

# Rolling metrics
df_rolling = df.with_columns([
    pl.col("returns").metrics.rolling_volatility().alias("rolling_vol"),
    pl.col("returns").metrics.rolling_sharpe().alias("rolling_sharpe"),
])
```

---

## Null Handling

NanuQuant follows QuantStats/pandas conventions by dropping null values before calculations:

```python
# Nulls are dropped automatically
returns_with_nulls = pl.Series([0.01, None, -0.02, None, 0.015])
sharpe = nq.sharpe(returns_with_nulls)  # Calculates using [0.01, -0.02, 0.015]
```

---

## Available Metrics

### Returns (12)
`comp`, `cagr`, `avg_return`, `avg_win`, `avg_loss`, `best`, `worst`, `win_rate`, `payoff_ratio`, `profit_factor`, `consecutive_wins`, `consecutive_losses`

### Risk (7)
`volatility`, `var`, `cvar`, `max_drawdown`, `to_drawdown_series`, `ulcer_index`, `downside_deviation`

### Performance (16)
`sharpe`, `sortino`, `calmar`, `omega`, `gain_to_pain_ratio`, `ulcer_performance_index`, `kelly_criterion`, `tail_ratio`, `common_sense_ratio`, `risk_return_ratio`, `recovery_factor`, `greeks`, `information_ratio`, `r_squared`, `treynor_ratio`, `benchmark_correlation`

### Distribution (12)
`skewness`, `kurtosis`, `jarque_bera`, `shapiro_wilk`, `outlier_win_ratio`, `outlier_loss_ratio`, `expected_return`, `geometric_mean`, `outliers`, `outliers_iqr`, `remove_outliers`, `remove_outliers_iqr`

### Rolling (5)
`rolling_volatility`, `rolling_sharpe`, `rolling_sortino`, `rolling_beta`, `rolling_greeks`

### Trading (12)
`exposure`, `ghpr`, `rar`, `cpc_index`, `serenity_index`, `risk_of_ruin`, `adjusted_sortino`, `smart_sharpe`, `smart_sortino`, `sqn`, `expectancy`, `k_ratio`

### Institutional (15+)
`probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `minimum_track_record_length`, `arch_effect_test`, `garch_volatility`, `cornish_fisher_var`, `modified_var`, `entropic_var`, `marginal_contribution_to_risk`, `ledoit_wolf_covariance`, `absorption_ratio`, `lower_tail_dependence`, `implementation_shortfall`, `market_impact_estimate`

---

## Known Differences from QuantStats

NanuQuant intentionally differs from QuantStats in some areas for improved consistency:

| Aspect | NanuQuant | Rationale |
|--------|-----------|-----------|
| **CAGR/Calmar** | Periods-based calculation | Works with any time series, not just datetime-indexed |
| **Treynor Ratio** | CAGR / Beta | Standard academic definition |
| **Omega Ratio** | Correct implementation | Fixes bug in some QuantStats versions |
| **CPC Index** | PF x WR x PR | Standard trading literature formula |
| **Smart Sharpe** | Lo (2002) adjustment | Established academic reference |

See [Mathematics](docs/mathematics.md) for detailed formula explanations.

---

## Testing and Validation

NanuQuant is rigorously tested:

- **Differential tests** against QuantStats for consistency
- **Real market data** fixtures (SPY, QQQ, BND)
- **Edge case coverage** (empty data, nulls, all positive/negative)
- **Statistical tests** validation
- **Type checking** with strict mypy

```bash
# Run all tests
pytest

# Run differential tests
pytest tests/test_vs_quantstats.py -v

# Run integration tests with real market data
pytest -m integration

# Type checking
mypy nanuquant
```

See [Testing Methodology](docs/testing.md) for details.

---

## Development

```bash
# Clone and install
git clone https://github.com/launchstack-dev/nanuquant.git
cd nanuquant
pip install -e ".[dev]"

# Run tests
pytest

# Type check
mypy nanuquant
```

---

## Important Notices

### Not Financial Advice

NanuQuant is designed for:
- **Educational purposes** - Learning about quantitative finance
- **Research** - Academic and professional analysis
- **Strategy development** - Initial screening of trading ideas

NanuQuant is **NOT** designed for:
- Automated trading without human oversight
- Sole basis for investment decisions
- Regulatory compliance calculations

### Limitations

- **Past performance** does not predict future results
- **Backtests** suffer from survivorship bias, look-ahead bias, and overfitting
- **Statistical models** assume conditions that may not hold in real markets
- **Transaction costs** and market impact are not automatically included

Always consult a qualified financial professional before making investment decisions.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Links

- [GitHub Repository](https://github.com/launchstack-dev/nanuquant)
- [PyPI Package](https://pypi.org/project/nanuquant/)
- [Issue Tracker](https://github.com/launchstack-dev/nanuquant/issues)
- [Changelog](CHANGELOG.md)
