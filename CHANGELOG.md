# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-18

### Added

- **Core Metrics Module** - 21 fundamental quantitative finance metrics
  - Returns: `cagr`, `total_return`, `avg_return`, `win_rate`, `best`, `worst`
  - Risk: `volatility`, `max_drawdown`, `var`, `cvar`
  - Performance: `sharpe`, `sortino`, `calmar`, `omega`, `gain_to_pain_ratio`, `tail_ratio`, `common_sense_ratio`, `cpc_index`, `outlier_win_ratio`, `outlier_loss_ratio`
  - Distribution: `skewness`, `kurtosis`
  - Rolling: `rolling_sharpe`, `rolling_sortino`, `rolling_volatility`, `rolling_beta`

- **Advanced Trading Metrics** - 12 trading-specific metrics
  - `sqn` - System Quality Number
  - `expectancy` - Average expected return per trade
  - `k_ratio` - Equity curve consistency
  - `profit_factor` - Gross profit / gross loss ratio
  - `payoff_ratio` - Average win / average loss
  - `exposure` - Time in market
  - `risk_of_ruin` - Probability of account depletion
  - And more

- **Institutional Robustness Metrics**
  - `probabilistic_sharpe_ratio` (PSR) - Statistical significance of Sharpe ratio
  - `deflated_sharpe_ratio` (DSR) - Multiple testing adjusted Sharpe ratio

- **Reports Module**
  - `compute_stats` - Aggregate all metrics into a dictionary
  - `generate_tearsheet` - HTML tearsheet report generation
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
