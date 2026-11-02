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

NanuQuant is a drop-in QuantStats replacement built natively on Polars. Zero Pandas dependency in production, a 33x median speedup, missing data handled as missing rather than as zero, and institutional-grade metrics that go well beyond standard libraries.

### Performance: 33x Median Speedup Over QuantStats

Benchmarked on 24 metrics on real market data (SPY, 8,298 observations):

| Metric | NanuQuant | QuantStats | Speedup |
|--------|----------:|----------:|--------:|
| avg_return | 0.010 ms | 0.471 ms | **45x** |
| win_rate | 0.011 ms | 0.560 ms | **50x** |
| kelly_criterion | 0.038 ms | 2.453 ms | **64x** |
| sharpe | 0.011 ms | 0.370 ms | **33x** |
| sortino | 0.156 ms | 0.547 ms | **4x** |
| max_drawdown | 0.267 ms | 0.639 ms | **2x** |
| gain_to_pain | 0.011 ms | 1.300 ms | **119x** |

> Median **33x** across the 24 metrics, from 1.3x (kurtosis) to 119x. Timings vary by hardware; see the [full benchmark results](https://launchstack-dev.github.io/nanuquant/benchmarks/).

### Calculation Accuracy: Verified Against QuantStats

Every metric that is meant to match QuantStats is checked against [quantstats-lumi](https://github.com/Lumiwealth/quantstats_lumi) at a relative tolerance of **1e-9**:

- **Data:** the full SPY (1993–2026, 8,298 days), QQQ and BND histories, plus SPY with every seventh day flat, so zero-return handling is exercised.
- **Coverage:**
  - 38 single-series metrics, from `comp` and `sharpe` to `serenity_index`, `cpc_index`, `smart_sortino` and `risk_of_ruin`;
  - alpha, beta, information ratio and R² against QQQ;
  - rolling volatility, Sharpe and Sortino.
- **Deliberate differences** (below) are pinned to their exact relationship with QuantStats rather than skipped. For example, drawdown metrics on a series that starts with a loss must equal QuantStats run with the starting capital as an explicit first point.

| Metric (SPY) | NanuQuant | QuantStats |
|--------|----------:|----------:|
| sharpe | 0.639751 | 0.639751 |
| sortino | 0.910049 | 0.910049 |
| volatility | 0.186220 | 0.186220 |
| max_drawdown | -0.551894 | -0.551894 |
| var (95%) | -0.018823 | -0.018823 |
| skewness | -0.010854 | -0.010854 |
| kurtosis | 11.8540 | 11.8540 |

**Missing data is missing, never zero.** QuantStats fills missing returns with 0, which reads a missing day as a flat day. In NanuQuant:
- a metric on a series with gaps equals the metric on the series with the gaps removed;
- a paired metric uses only the periods where both series are present;
- a positional output (drawdown series, rolling metrics) keeps its length and is null at the gap.

This is pinned by its own test suite (`tests/test_missing_data.py`).

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
| **Risk** | `volatility`, `var`, `cvar`, `max_drawdown`, `to_drawdown_series`, `drawdown_details`, `ulcer_index`, `downside_deviation` |
| **Performance** | `sharpe`, `sortino`, `calmar`, `omega`, `gain_to_pain_ratio`, `ulcer_performance_index`, `kelly_criterion`, `tail_ratio`, `common_sense_ratio`, `risk_return_ratio`, `recovery_factor`, `greeks`, `information_ratio`, `r_squared`, `treynor_ratio`, `benchmark_correlation` |
| **Distribution** | `skewness`, `kurtosis`, `jarque_bera`, `shapiro_wilk`, `outlier_win_ratio`, `outlier_loss_ratio`, `expected_return`, `geometric_mean` |
| **Rolling** | `rolling_volatility`, `rolling_sharpe`, `rolling_sortino`, `rolling_beta`, `rolling_greeks` |

### Advanced Trading

`exposure`, `ghpr`, `rar`, `cpc_index`, `serenity_index`, `risk_of_ruin`, `adjusted_sortino`, `smart_sharpe`, `smart_sortino`, `sqn`, `expectancy`, `k_ratio`

### Institutional

`probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `minimum_track_record_length`, `arch_effect_test`, `garch_volatility`, `cornish_fisher_var`, `modified_var`, `entropic_var`, `marginal_contribution_to_risk`, `ledoit_wolf_covariance`, `absorption_ratio`, `lower_tail_dependence`, `implementation_shortfall`, `market_impact_estimate`

### Indicators

`nanuquant.indicators`: price indicators as Polars expressions. Each one takes a `pl.Series` or a `pl.Expr` (any number of columns, lazily), and there is a `.ind` namespace:
- `current_price`, `sma`, `ema`, `rsi`;
- `cumulative_return`, `max_drawdown`, `current_drawdown`;
- `std_dev_return`, `moving_average_return`.

Platform conventions are explicit arguments (`percent=`, `annualize=`, `rsi(seed=)`), so a strategy engine can pin, for example, Composer's.

---

## Known Differences from QuantStats

NanuQuant intentionally differs from QuantStats where we believe the alternative is more correct or more general. Each difference is pinned in `tests/test_quantstats_parity.py` by its exact relationship to QuantStats:

| Aspect | NanuQuant | QuantStats |
|--------|-----------|------------|
| **Missing data** | Null/NaN dropped (pairwise for two series; positional outputs are null at the gap) | Filled with 0, so a missing day reads as a flat one |
| **Drawdown** (and Calmar, recovery factor, ulcer index/UPI, serenity) | Measured from the starting capital | Peak starts at the first period's wealth, so a series that has only lost since inception shows no drawdown. Identical once the first return is non-negative |
| **CAGR, Calmar, RAR** | Years = periods / `periods_per_year` | Years from the date index (within 0.3% on daily data). Works with any series, dated or not |
| **Treynor ratio** | CAGR / beta | Total multi-year return / beta |
| **Exposure** | Exact share of non-zero periods | Rounded up to a whole percent |
| **Risk-free rate** | Per period = `rf / periods` | `(1 + rf)^(1/periods) - 1`; about 1% apart on Sharpe at rf = 4% |
| **Rolling metrics** | Annualize with `periods_per_year` (252), like the core metrics | Default 365; pass `periods_per_year=365` to reproduce |
| **Omega** | QuantStats' formula; +inf with no losses | quantstats-lumi's `omega` fails on a Series |
| **Kelly criterion** | -inf with no winning periods, 0 with no non-zero periods | NaN |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Installation](https://launchstack-dev.github.io/nanuquant/installation/) | Setup and configuration |
| [Quick Start](https://launchstack-dev.github.io/nanuquant/quickstart/) | Get started in minutes |
| [Core API](https://launchstack-dev.github.io/nanuquant/api/core/) | Returns, risk, performance metrics |
| [Advanced API](https://launchstack-dev.github.io/nanuquant/api/advanced/) | Trading system metrics |
| [Institutional API](https://launchstack-dev.github.io/nanuquant/api/institutional/) | PSR, DSR, GARCH, VaR extensions |
| [Indicators API](https://launchstack-dev.github.io/nanuquant/api/indicators/) | Price indicators as Polars expressions |
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

## Acknowledgments

NanuQuant owes a debt of gratitude to [QuantStats](https://github.com/ranaroussi/quantstats) by Ran Aroussi, which pioneered accessible quantitative analytics in Python. NanuQuant's API design, metric definitions, and differential test suite are all built on the foundation QuantStats established. Our benchmarks run against [quantstats-lumi](https://github.com/Lumiwealth/quantstats_lumi), the actively maintained community fork by Lumiwealth. We benchmark against these libraries not as criticism, but as respect — you can't improve on something you don't deeply understand.

---

## License

MIT License. See [LICENSE](LICENSE).
