# NanuQuant Documentation

> **DISCLAIMER**: NanuQuant is for **educational and research purposes only**. Nothing in this library constitutes financial advice. See [DISCLAIMER.md](../DISCLAIMER.md).

Welcome to the NanuQuant documentation. NanuQuant is a high-performance quantitative finance library built natively on Polars.

## About the Name

In Inuktitut, the language of the Inuit people, **"nanuq"** means **polar bear**. Since NanuQuant is built entirely on Polars for maximum speed and efficiency, we chose this name to honor the Inuit people while celebrating our foundation on the Polars ecosystem.

---

## Getting Started

- [Installation Guide](installation.md) - How to install NanuQuant
- [Quick Start](quickstart.md) - Get up and running in minutes

---

## API Reference

Detailed documentation for all NanuQuant modules:

### Core Metrics
- [Core API Reference](api/core.md) - Returns, risk, performance, distribution, and rolling metrics

### Advanced Metrics
- [Advanced Trading Metrics](api/advanced.md) - SQN, expectancy, K-ratio, Smart Sharpe

### Institutional Metrics
- [Institutional Analytics](api/institutional.md) - PSR, DSR, GARCH, VaR extensions, portfolio analytics

---

## Understanding the Metrics

- [Mathematical Foundations](mathematics.md) - Detailed formulas and theory behind each metric
- [Use Cases](use-cases.md) - Practical examples with important caveats

---

## Quality Assurance

- [Testing Methodology](testing.md) - How NanuQuant is validated and tested

---

## Important Information

- [Full Disclaimer](../DISCLAIMER.md) - Legal notices and limitations
- [Changelog](../CHANGELOG.md) - Version history

---

## Quick Reference

### Core Functions

```python
import nanuquant as nq

# Performance
nq.sharpe(returns)
nq.sortino(returns)
nq.calmar(returns)
nq.omega(returns)

# Risk
nq.volatility(returns)
nq.max_drawdown(returns)
nq.var(returns)
nq.cvar(returns)

# Returns
nq.cagr(returns)
nq.comp(returns)
nq.win_rate(returns)

# Distribution
nq.skewness(returns)
nq.kurtosis(returns)
```

### Institutional Functions

```python
from nanuquant import institutional

# Robustness
institutional.probabilistic_sharpe_ratio(returns)
institutional.deflated_sharpe_ratio(returns, n_trials=100)

# Volatility
institutional.arch_effect_test(returns)
institutional.garch_volatility(returns)

# Advanced VaR
institutional.cornish_fisher_var(returns)

# Portfolio
institutional.marginal_contribution_to_risk(returns_df, weights)
institutional.ledoit_wolf_covariance(returns_df)
```

### DataFrame Namespace

```python
import polars as pl
import nanuquant as nq

df = pl.DataFrame({"returns": [...]})

# Use .metrics namespace
df.select(pl.col("returns").metrics.sharpe())
df.with_columns(pl.col("returns").metrics.rolling_sharpe())
```

---

## Need Help?

- Check the [GitHub Issues](https://github.com/launchstack-dev/nanuquant/issues)
- Review the [Use Cases](use-cases.md) for examples

---

*Remember: NanuQuant is for educational purposes only. Always consult a qualified financial professional before making investment decisions.*
