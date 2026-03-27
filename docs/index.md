# NanuQuant

> **DISCLAIMER**: NanuQuant is for **educational and research purposes only**. Nothing in this library constitutes financial advice. See [DISCLAIMER](https://github.com/launchstack-dev/nanuquant/blob/main/DISCLAIMER.md).

NanuQuant is a high-performance quantitative finance library built natively on Polars. It provides 60+ core metrics, 12 advanced trading metrics, and 15+ institutional-grade analytics — all without a single Pandas dependency in production.

## About the Name

In Inuktitut, the language of the Inuit people, **"nanuq"** means **polar bear**. Since NanuQuant is built entirely on Polars for maximum speed and efficiency, we chose this name to honor the Inuit people while celebrating our foundation on the Polars ecosystem.

---

## Performance Highlights

NanuQuant is benchmarked against QuantStats on every release:

- **13x median speedup** over QuantStats across 24 metrics
- **56x peak speedup** on kelly_criterion
- **21/24 metrics** match QuantStats within 1e-8 tolerance (3 are intentional improvements)

See the full [Benchmarks](benchmarks.md) page for detailed timing and accuracy tables.

---

## Getting Started

- [Installation Guide](installation.md) — Setup and configuration
- [Quick Start](quickstart.md) — Get up and running in minutes

---

## API Reference

- [Core Metrics](api/core.md) — Returns, risk, performance, distribution, rolling
- [Advanced Trading Metrics](api/advanced.md) — SQN, expectancy, K-ratio, Smart Sharpe
- [Institutional Analytics](api/institutional.md) — PSR, DSR, GARCH, VaR extensions, portfolio analytics

---

## Understanding the Metrics

- [Mathematical Foundations](mathematics.md) — Formulas and theory behind each metric
- [Use Cases](use-cases.md) — Practical examples with important caveats

---

## Quality Assurance

- [Benchmarks](benchmarks.md) — Performance and accuracy comparison vs QuantStats
- [Testing Methodology](testing.md) — How NanuQuant is validated

---

## Quick Reference

```python
import polars as pl
import nanuquant as nq

returns = pl.Series("returns", [0.01, -0.02, 0.03, 0.01, -0.01, 0.02])

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

# Polars namespace
df = pl.DataFrame({"returns": [0.01, -0.02, 0.03, 0.01, -0.01]})
df.select(pl.col("returns").metrics.sharpe())
```

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

---

## Contributing

See [CONTRIBUTING](https://github.com/launchstack-dev/nanuquant/blob/main/CONTRIBUTING.md) for development setup, coding standards, and PR guidelines.

---

*NanuQuant is for educational purposes only. Always consult a qualified financial professional before making investment decisions.*
