# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`probabilistic_sharpe_ratio`, `deflated_sharpe_ratio` and `minimum_track_record_length` overstated significance.** The Bailey & López de Prado standard error is defined on the per-observation Sharpe ratio, but the annualized ratio was plugged in. On SPY this gave PSR = 1.0 and a minimum track record of 18 days; it is now 0.99987 and 1,677 observations (about 6.7 years). `benchmark_sr`, `target_sr` and `var_sharpe` remain **annualized** inputs and are documented as such. `minimum_track_record_length` now returns `float` (`inf` when the target can't be reached), and is exported at the top level along with `newey_west_se`. Tests pin all three to an independent implementation of the papers.
- **Metrics documented as matching QuantStats now do, checked at `rtol=1e-9` on the full SPY, QQQ and BND histories** (`tests/test_quantstats_parity.py`). The previous integration tests allowed 3% error, and some only checked that the result was finite. **These change values:**
  - `win_rate`, `avg_return` and `kelly_criterion` exclude zero-return periods, like QuantStats. Pass `exclude_zeros=False` to `win_rate` or `avg_return` to count them.
  - `expected_return` is the geometric mean (it was the arithmetic mean).
  - `outlier_win_ratio` / `outlier_loss_ratio` use QuantStats' definition: a tail quantile of all returns over the mean up (down) return. `outlier_loss_ratio` now defaults to `quantile=0.01`.
  - `adjusted_sortino` is `sortino / sqrt(2)`. It was a skew/kurtosis adjustment that QuantStats does not use.
  - `cpc_index` is `profit_factor * win_rate * payoff_ratio`.
  - `serenity_index` is `(sum(r) - rf) / (ulcer_index * pitfall)`, and it takes `risk_free_rate`.
  - `smart_sharpe` / `smart_sortino` use QuantStats' full autocorrelation penalty, not a lag-1 `2*rho` approximation.
  - `risk_of_ruin` uses QuantStats' formula, computed in log space. It previously raised `OverflowError` on long losing series.
- **Edge cases now report the mathematical limit instead of a masking value:**
  - `kelly_criterion` with no winning periods is `-inf`;
  - `omega` with no losses is `inf`;
  - `geometric_mean` / `ghpr` after a total loss are `-1`, and `NaN` for wealth below zero. All of these were `0.0`.
- Docstrings corrected where nanuquant deliberately differs from QuantStats, with tests pinning each relationship:
  - `treynor_ratio` uses annualized return (QuantStats divides total return by beta);
  - `exposure` is exact (QuantStats rounds up to 2 decimals);
  - `cagr` and `rar` count periods (QuantStats uses calendar dates).
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
