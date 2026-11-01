# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Missing data is now handled consistently, and never as a zero return.** A missing observation is a Polars `null` or a float `NaN`. Previously most metrics called `validate_returns()` and discarded its cleaned result, so nulls were still counted by `len()`, fed to numpy and scipy, and broke `win_rate`, `cagr`, `sortino`, `skewness`, `kurtosis`, `geometric_mean`, VaR and others. QuantStats, for comparison, fills missing returns with 0. Now:
  - single-series metrics drop missing observations;
  - paired metrics (`greeks`, `information_ratio`, `r_squared`, `treynor_ratio`, `benchmark_correlation`, the tail-dependence and regime correlations, `compare`) use only the rows where both series are present (`align_pair`);
  - dated outputs (`yearly_returns`, `monthly_returns`, `distribution`, `drawdown_details`) drop a missing return together with its date (`align_dates`), and reject a `dates` series of a different length;
  - positional outputs (`to_drawdown_series`, rolling metrics) keep their length and are null where the input is missing. For rolling metrics, any window containing a missing value is null.
- `newey_west_se` now validates its input.
- `distribution()`, `monthly_returns()`, `yearly_returns()` and `drawdown_details()` return periods in a deterministic, chronological order (an unordered `group_by` made them random between calls).

### Added

- **Benchmark suite** (`benchmarks/run_benchmarks.py`) — automated performance and accuracy comparison against QuantStats
- **Benchmark documentation** (`docs/benchmarks.md`) — full timing tables and calculation audit results across synthetic and real market data
- **CONTRIBUTING.md** — development setup, coding standards, PR guidelines
- Benchmarks page added to MkDocs navigation

### Changed

- **README.md** — rewritten for open-source readiness with benchmark highlights, accuracy audit tables, and cleaner structure
- **docs/index.md** — added performance highlights and benchmarks section
- Updated copyright year range to 2024-2026

## [0.1.0] - 2024-01-18

### Added

- **Core Metrics Module** — 21 fundamental quantitative finance metrics
  - Returns: `cagr`, `total_return`, `avg_return`, `win_rate`, `best`, `worst`
  - Risk: `volatility`, `max_drawdown`, `var`, `cvar`
  - Performance: `sharpe`, `sortino`, `calmar`, `omega`, `gain_to_pain_ratio`, `tail_ratio`, `common_sense_ratio`, `cpc_index`, `outlier_win_ratio`, `outlier_loss_ratio`
  - Distribution: `skewness`, `kurtosis`
  - Rolling: `rolling_sharpe`, `rolling_sortino`, `rolling_volatility`, `rolling_beta`

- **Advanced Trading Metrics** — 12 trading-specific metrics
  - `sqn` — System Quality Number
  - `expectancy` — Average expected return per trade
  - `k_ratio` — Equity curve consistency
  - `profit_factor` — Gross profit / gross loss ratio
  - `payoff_ratio` — Average win / average loss
  - `exposure` — Time in market
  - `risk_of_ruin` — Probability of account depletion
  - And more

- **Institutional Robustness Metrics**
  - `probabilistic_sharpe_ratio` (PSR) — Statistical significance of Sharpe ratio
  - `deflated_sharpe_ratio` (DSR) — Multiple testing adjusted Sharpe ratio

- **Reports Module**
  - `compute_stats` — Aggregate all metrics into a dictionary
  - `generate_tearsheet` — HTML tearsheet report generation
  - Customizable metric selection and formatting

- **Configuration System**
  - Global defaults for risk-free rate, periods per year, and more
  - Context manager for temporary configuration changes

- **Full Type Hint Support**
  - PEP 561 compliant with `py.typed` marker
  - Strict mypy configuration

### Technical Details

- Native Polars implementation with zero pandas dependency in production
- Lazy evaluation support for large datasets
- Comprehensive test suite with differential tests against QuantStats
- Python 3.10+ required
