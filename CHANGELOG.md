# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Drawdown is measured from the starting capital.** **Breaking** for series that start with a loss: the running peak starts at 1.0 instead of the first period's wealth (the QuantStats convention), so a series that has only lost since inception now shows its drawdown: two -5% days give -9.75%, where QuantStats reports -5%, and `[-10%, +5%]` gives -5.5% instead of 0. This affects `to_drawdown_series`, `max_drawdown`, `drawdown_details` and everything built on them (Calmar, recovery factor, ulcer index, ulcer performance index, serenity); from a non-negative first return the two conventions agree exactly. `drawdown_details` reports such an episode with a null `start` (the starting capital) and counts its `length` from there. Wealth at or below zero (a return below -100%) is ruin: the drawdown is -100% from then on.
- **One VaR sign convention.** `cornish_fisher_var`, `modified_var`, `historical_var`, `parametric_var` and `entropic_var` now return a signed return (a loss is negative), the same as `var`, `cvar` and QuantStats. They used to return a positive loss. **Breaking:** negate existing callers.
- **The risk-free rate is converted to a per-period rate arithmetically (`rf / periods`), not geometrically as in QuantStats.** Documented as a deliberate difference (Sharpe and Sortino with rf = 4% differ by about 1%) and pinned exactly in the parity tests.
- **Rolling metrics on a series shorter than the window return all-null output of the input's length** instead of an empty Series, so they work inside the `.metrics` expression namespace (`map_batches` needs the frame's height).
- **Rolling metrics annualize with the configured `periods_per_year` (252 by default)**, like the core metrics, so `rolling_sharpe` and `sharpe` are on the same scale. QuantStats' rolling default is 365; pass `periods_per_year=365` to reproduce it. Strict rolling parity tests were added.
- **`garch_volatility` no longer silently fakes an estimate.** Without the optional `arch` package it falls back to fixed alpha=0.10, beta=0.85; it now issues a `UserWarning`, and `GARCHResult.method` reports `"mle"` or `"variance_targeting"`. `GARCHResult` therefore has eight fields; code that unpacks it positionally into seven names must add one.
- **Portfolio and systemic functions drop incomplete rows instead of zero-filling them:** `marginal_contribution_to_risk`, `ledoit_wolf_covariance`, `portfolio_volatility`, `absorption_ratio`. Zero-filling biased covariances toward zero. They raise if fewer than two complete rows remain (`absorption_ratio`: fewer complete rows than assets).
- **Trade equity curves:**
  - with `prices`, the curve follows the price dates (trading days) of the traded symbols plus every trade entry and exit date, instead of every calendar day, and annualizes at 252, or 365 when its dates include weekends like a 7-day (crypto) feed (for both `aggregation="equity"` and `"D"`);
  - without prices, it stays on calendar days and now reports `periods_per_year=365` (it reported 252), as does a daily `aggregation="D"` series filled to every calendar day with `fill_gaps=True`;
  - price lookups use binary search instead of scanning every price for every position and day.
  - a price row without a date is ignored (it raised `TypeError`);
  - a missing close (null or NaN) values a position at the last known close; it used to fall back to the entry price for that day, booking a spurious loss and gain.
- The HTML report no longer crashes with `OverflowError` on an infinite metric (e.g. `profit_factor` with no losing period, `kelly_criterion` with no winning one); it prints `inf`.
- HTML report footer and `py.typed` still said "polars-metrics"/"polars_metrics".
- **`probabilistic_sharpe_ratio`, `deflated_sharpe_ratio` and `minimum_track_record_length` overstated significance.** The Bailey & López de Prado standard error is defined on the per-observation Sharpe ratio, but the annualized ratio was plugged in. On SPY this gave PSR = 1.0 and a minimum track record of 18 days; it is now 0.99987 and 1,677 observations (about 6.7 years). `benchmark_sr`, `target_sr` and `var_sharpe` remain **annualized** inputs and are documented as such. `minimum_track_record_length` now returns `float` (`inf` when the target can't be reached; a constant series takes its Sharpe ratio as +inf, -inf or 0 by the sign of its return instead of returning its own length), and is exported at the top level along with `newey_west_se`. Tests pin all three to an independent implementation of the papers.
- **Metrics documented as matching QuantStats now do, checked at `rtol=1e-9` on the full SPY, QQQ and BND histories** (`tests/test_quantstats_parity.py`). The previous integration tests allowed 3% error, and some only checked that the result was finite. **These change values:**
  - `win_rate`, `avg_return` and `kelly_criterion` exclude zero-return periods, like QuantStats. Pass `exclude_zeros=False` to `win_rate` or `avg_return` to count them.
  - `expected_return` is the geometric mean (it was the arithmetic mean).
  - `outlier_win_ratio` / `outlier_loss_ratio` use QuantStats' definition: a tail quantile of all returns over the mean up (down) return. `outlier_loss_ratio` now defaults to `quantile=0.01`.
  - `adjusted_sortino` is `sortino / sqrt(2)`. It was a skew/kurtosis adjustment that QuantStats does not use.
  - `cpc_index` is `profit_factor * win_rate * payoff_ratio`.
  - `serenity_index` is `(sum(r) - rf) / (ulcer_index * pitfall)`, and it takes `risk_free_rate`.
  - `smart_sharpe` / `smart_sortino` use QuantStats' full autocorrelation penalty, not a lag-1 `2*rho` approximation.
  - `risk_of_ruin` uses QuantStats' formula, computed in log space. It previously raised `OverflowError` on long losing series. Its `target_fraction` argument, which was never used, is removed.
- **Edge cases now report the mathematical limit instead of a masking value:**
  - `kelly_criterion` with no winning periods is `-inf`;
  - `omega` with no losses is `inf`;
  - `geometric_mean` / `ghpr` after a total loss are `-1`, and `NaN` for wealth below zero. All of these were `0.0`. They are computed from the mean log growth, so a long series no longer underflows to a spurious total loss (or overflows to `inf`).
- Docstrings corrected where nanuquant deliberately differs from QuantStats, with tests pinning each relationship:
  - `treynor_ratio` uses annualized return (QuantStats divides total return by beta);
  - `exposure` is exact (QuantStats rounds up to 2 decimals);
  - `cagr` and `rar` count periods (QuantStats uses calendar dates).
- **Missing data is now handled consistently, and never as a zero return.** A missing observation is a Polars `null` or a float `NaN`. Previously most metrics called `validate_returns()` and discarded its cleaned result, so nulls were still counted by `len()`, fed to numpy and scipy, and broke `win_rate`, `cagr`, `sortino`, `skewness`, `kurtosis`, `geometric_mean`, VaR and others. QuantStats, for comparison, fills missing returns with 0. Now:
  - single-series metrics drop missing observations;
  - paired metrics (`greeks`, `information_ratio`, `r_squared`, `treynor_ratio`, `benchmark_correlation`, the tail-dependence and regime correlations, `compare`) use only the rows where both series are present (`align_pair`);
  - dated outputs (`yearly_returns`, `monthly_returns`, `distribution`, `drawdown_details`) drop a missing return together with its date (`align_dates`), and reject a `dates` series of a different length. Without `dates`, each return keeps its own position: the synthetic daily dates are assigned before missing returns are dropped, and `drawdown_details` reports indices into the input series;
  - positional outputs (`to_drawdown_series`, `cumulative_returns` / `equity_curve` / `compound_returns`, rolling metrics) keep their length and are null where the input is missing. For rolling metrics, any window containing a missing value is null; a cumulative curve compounds on past the gap (a `NaN` used to turn the rest of the curve into `NaN`);
  - a series with no value at all (dtype `Null`) is all-missing rather than a `TypeError`.
- `newey_west_se` now validates its input.
- `drawdown_details` uses `to_drawdown_series`, so a total loss in the first period is reported as a -100% drawdown (it was skipped as NaN).
- `distribution()`, `monthly_returns()`, `yearly_returns()` and `drawdown_details()` return periods in a deterministic, chronological order (an unordered `group_by` made them random between calls).
- `distribution()` keys weekly buckets by ISO year, so the last days of December that fall in ISO week 1 of the next year are no longer merged into the first week of their own year (a 9-day "week").
- `parametric_var` and `cornish_fisher_var` of a zero-volatility series are the constant return itself (it was 0.0, which read as "no loss" under the signed convention).
- `probabilistic_sharpe_ratio` and `deflated_sharpe_ratio` treat a constant series as zero volatility even when rounding leaves a tiny nonzero standard deviation (they returned NaN), as `minimum_track_record_length` already did.

### Added

- **Indicators subpackage** (`nanuquant.indicators`) — price indicators as Polars expressions: `current_price`, `sma`, `ema`, `rsi` (Wilder), `cumulative_return`, `max_drawdown` (worst-in-window), `current_drawdown` (all-time or rolling peak), `std_dev_return`, `moving_average_return`. Each accepts a `pl.Series` or `pl.Expr` and returns the same kind; platform conventions (`percent=`, `annualize=`, peak window) are explicit arguments
- **`.ind` expression namespace** — `pl.col("SPY").ind.rsi(10)` alongside the existing `.metrics` namespace
- **Indicators API docs** (`docs/api/indicators.md`), including the Composer-compatible settings
- Differential tests pinning every indicator to a frozen reference implementation on SPY/QQQ/BND history and synthetic inception-gap and flat-window series
- **Benchmark suite** (`benchmarks/run_benchmarks.py`) — automated performance and accuracy comparison against QuantStats
- **Benchmark documentation** (`docs/benchmarks.md`) — full timing tables and calculation audit results across synthetic and real market data
- **CONTRIBUTING.md** — development setup, coding standards, PR guidelines
- Benchmarks page added to MkDocs navigation

### Changed

- **Minimum Polars version raised to 1.21** — the indicators use the `min_samples` window argument introduced in Polars 1.21
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
