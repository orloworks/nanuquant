# NanuQuant

### Institutional quantitative analytics at Polars speed.

[![PyPI](https://img.shields.io/pypi/v/nanuquant)](https://pypi.org/project/nanuquant/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://launchstack-dev.github.io/nanuquant/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/launchstack-dev/nanuquant/actions/workflows/ci.yml/badge.svg)](https://github.com/launchstack-dev/nanuquant/actions/workflows/ci.yml)

---

> **DISCLAIMER**: NanuQuant is for **educational and research purposes only**. Nothing in this library constitutes financial advice. Past performance does not guarantee future results. See [DISCLAIMER.md](DISCLAIMER.md) for full legal notices.

---

## About the Name

In Inuktitut, the language of the Inuit people, **"nanuq"** means **polar bear**. Since NanuQuant is built entirely on [Polars](https://pola.rs/) for maximum speed and efficiency, we chose this name to honor the Inuit people while celebrating our foundation on the Polars ecosystem.

---

## Why NanuQuant?

NanuQuant is a drop-in QuantStats replacement built natively on Polars. Zero Pandas dependency in production, 13x median speedup, and institutional-grade metrics that go well beyond standard libraries.

### Performance: 3x–56x Faster Than QuantStats

Benchmarked on 24 metrics across synthetic and real market data (SPY, 8,298 observations):

| Metric | NanuQuant | QuantStats | Speedup |
|--------|----------:|----------:|--------:|
| avg_return | 0.017 ms | 0.386 ms | **22x** |
| win_rate | 0.019 ms | 0.406 ms | **22x** |
| kelly_criterion | 0.030 ms | 1.686 ms | **56x** |
| sharpe | 0.023 ms | 0.301 ms | **13x** |
| sortino | 0.096 ms | 0.380 ms | **4x** |
| max_drawdown | 0.136 ms | 0.473 ms | **4x** |
| gain_to_pain | 0.023 ms | 0.654 ms | **28x** |

> Median **13x** speedup across all metrics. See the [full benchmark results](https://launchstack-dev.github.io/nanuquant/benchmarks/).

### Calculation Accuracy: Verified Against QuantStats

Every metric is differentially tested against QuantStats to verify correctness. On synthetic data, **21 of 24 metrics match within 1e-8 tolerance** — the 3 differences are intentional improvements (documented below).

| Metric | NanuQuant | QuantStats | Rel. Diff | Status |
|--------|----------:|----------:|----------:|--------|
| sharpe | 0.639751 | 0.639751 | 0% | PASS |
| sortino | 0.910049 | 0.910049 | <0.001% | PASS |
| volatility | 0.186220 | 0.186220 | <0.001% | PASS |
| max_drawdown | -0.551894 | -0.551894 | <0.001% | PASS |
| var (95%) | -0.018823 | -0.018823 | <0.001% | PASS |
| skewness | -0.010854 | -0.010854 | 0% | PASS |
| kurtosis | 11.8540 | 11.8540 | <0.001% | PASS |

> Full audit tables for all metrics and dataset sizes in [Benchmarks](https://launchstack-dev.github.io/nanuquant/benchmarks/).

### Beyond QuantStats

NanuQuant includes 15+ institutional-grade metrics not available in QuantStats:

- **Probabilistic Sharpe Ratio** — Is your Sharpe statistically significant, or just noise?
- **Deflated Sharpe Ratio** — Adjust for multiple strategy testing (data snooping)
- **GARCH(1,1) Volatility** — Model volatility clustering and regime changes
- **Cornish-Fisher VaR** — Skewness/kurtosis-adjusted Value at Risk
- **Ledoit-Wolf Covariance** — Shrinkage estimator for portfolio optimization
- **Absorption Ratio** — Systemic risk measurement
- **Implementation Shortfall** — Execution quality analysis

---

## Installation

```bash
pip install nanuquant
```

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

# Create or load return data
returns = pl.Series("returns", [0.01, -0.02, 0.03, 0.01, -0.01, 0.02])

# Core metrics
print(f"Sharpe:       {nq.sharpe(returns):.4f}")
print(f"Sortino:      {nq.sortino(returns):.4f}")
print(f"Max Drawdown: {nq.max_drawdown(returns):.4%}")
print(f"Volatility:   {nq.volatility(returns):.4%}")
print(f"Win Rate:     {nq.win_rate(returns):.2%}")
```

### Polars Namespace Integration

```python
import polars as pl
import nanuquant as nq  # Registers .metrics namespace

df = pl.DataFrame({"returns": [0.01, -0.02, 0.015, -0.01, 0.02] * 50})

# Compute multiple metrics in one pass
metrics = df.select([
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

### Institutional Analysis

```python
from nanuquant import institutional

# Is your Sharpe ratio statistically significant?
psr = institutional.probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
print(f"Probability Sharpe > 0: {psr.psr:.2%}")

# Adjust for multiple strategy testing
dsr = institutional.deflated_sharpe_ratio(returns, n_trials=100)
print(f"Deflated Sharpe p-value: {dsr:.4f}")

# Detect volatility clustering
arch = institutional.arch_effect_test(returns)
if arch.has_arch_effects:
    garch = institutional.garch_volatility(returns)
    print(f"GARCH persistence: {garch.persistence:.4f}")
```

---

## Available Metrics (60+)

### Core

| Category | Metrics |
|----------|---------|
| **Returns** | `comp`, `cagr`, `avg_return`, `avg_win`, `avg_loss`, `best`, `worst`, `win_rate`, `payoff_ratio`, `profit_factor`, `consecutive_wins`, `consecutive_losses` |
| **Risk** | `volatility`, `var`, `cvar`, `max_drawdown`, `to_drawdown_series`, `ulcer_index`, `downside_deviation` |
| **Performance** | `sharpe`, `sortino`, `calmar`, `omega`, `gain_to_pain_ratio`, `ulcer_performance_index`, `kelly_criterion`, `tail_ratio`, `common_sense_ratio`, `risk_return_ratio`, `recovery_factor`, `greeks`, `information_ratio`, `r_squared`, `treynor_ratio`, `benchmark_correlation` |
| **Distribution** | `skewness`, `kurtosis`, `jarque_bera`, `shapiro_wilk`, `outlier_win_ratio`, `outlier_loss_ratio`, `expected_return`, `geometric_mean` |
| **Rolling** | `rolling_volatility`, `rolling_sharpe`, `rolling_sortino`, `rolling_beta`, `rolling_greeks` |

### Advanced Trading

`exposure`, `ghpr`, `rar`, `cpc_index`, `serenity_index`, `risk_of_ruin`, `adjusted_sortino`, `smart_sharpe`, `smart_sortino`, `sqn`, `expectancy`, `k_ratio`

### Institutional

`probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `minimum_track_record_length`, `arch_effect_test`, `garch_volatility`, `cornish_fisher_var`, `modified_var`, `entropic_var`, `marginal_contribution_to_risk`, `ledoit_wolf_covariance`, `absorption_ratio`, `lower_tail_dependence`, `implementation_shortfall`, `market_impact_estimate`

---

## Known Differences from QuantStats

NanuQuant intentionally differs from QuantStats where we believe the alternative is more correct or more general:

| Aspect | NanuQuant | Rationale |
|--------|-----------|-----------|
| **CAGR/Calmar** | Periods-based calculation | Works with any time series, not just datetime-indexed |
| **Treynor Ratio** | CAGR / Beta | Standard academic definition (annualized, not total return) |
| **Omega Ratio** | Correct implementation | Fixes bug in some QuantStats versions |
| **Smart Sharpe** | Lo (2002) adjustment | Established academic reference for autocorrelation penalty |

All differences are documented in tests and in the [benchmark accuracy tables](https://launchstack-dev.github.io/nanuquant/benchmarks/).

---

## Documentation

| Document | Description |
|----------|-------------|
| [Installation](https://launchstack-dev.github.io/nanuquant/installation/) | Setup and configuration |
| [Quick Start](https://launchstack-dev.github.io/nanuquant/quickstart/) | Get started in minutes |
| [Core API](https://launchstack-dev.github.io/nanuquant/api/core/) | Returns, risk, performance metrics |
| [Advanced API](https://launchstack-dev.github.io/nanuquant/api/advanced/) | Trading system metrics |
| [Institutional API](https://launchstack-dev.github.io/nanuquant/api/institutional/) | PSR, DSR, GARCH, VaR extensions |
| [Benchmarks](https://launchstack-dev.github.io/nanuquant/benchmarks/) | Performance and accuracy vs QuantStats |
| [Mathematics](https://launchstack-dev.github.io/nanuquant/mathematics/) | Formulas and theory |
| [Testing](https://launchstack-dev.github.io/nanuquant/testing/) | Validation methodology |

---

## Development

```bash
git clone https://github.com/launchstack-dev/nanuquant.git
cd nanuquant
pip install -e ".[dev]"

# Run tests
pytest

# Type check
mypy nanuquant

# Lint
ruff check nanuquant

# Run benchmarks
python benchmarks/run_benchmarks.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## Important Notices

NanuQuant is designed for **educational purposes** and **research**. It is NOT designed for automated trading without human oversight, as the sole basis for investment decisions, or for regulatory compliance calculations. Past performance does not predict future results. Always consult a qualified financial professional. See [DISCLAIMER.md](DISCLAIMER.md).

---

## License

MIT License. See [LICENSE](LICENSE).
