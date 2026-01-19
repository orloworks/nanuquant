# NanuQuant Project Overview

## Purpose
NanuQuant is a high-performance, native Polars library for quantitative finance analytics. It serves as a QuantStats replacement with zero Pandas dependency, providing institutional-grade metrics for backtesting, risk management, and trading strategy evaluation.

## Key Features
- **60+ quantitative metrics**: Returns, risk, performance, distribution, rolling metrics
- **Institutional analytics**: PSR, DSR, GARCH volatility, Cornish-Fisher VaR, Ledoit-Wolf covariance
- **Advanced trading metrics**: SQN, expectancy, K-ratio, Smart Sharpe/Sortino
- **Polars expression namespace**: `pl.col("returns").metrics.sharpe()`
- **Trade conversion**: Convert trade data to returns series
- **HTML report generation**: Comprehensive metric reports

## Tech Stack
- **Language**: Python 3.10+
- **Core dependencies**: polars (>=0.20.0), scipy (>=1.10.0), numpy (>=1.24)
- **Build system**: Hatchling
- **Package manager**: uv (recommended) or pip
- **Testing**: pytest with pytest-benchmark
- **Type checking**: mypy (strict mode)
- **Linting/Formatting**: ruff

## Project Structure
```
nanuquant/
├── __init__.py          # Main exports and namespace registration
├── config.py            # Global configuration (periods_per_year, ddof, etc.)
├── types.py             # Type definitions and constants
├── exceptions.py        # Custom exception classes
├── namespace.py         # Polars expression namespace (.metrics accessor)
├── core/                # Core metrics modules
│   ├── returns.py       # Return calculations (CAGR, comp, win_rate, etc.)
│   ├── risk.py          # Risk metrics (volatility, VaR, CVaR, max_drawdown)
│   ├── performance.py   # Performance ratios (Sharpe, Sortino, Calmar, etc.)
│   ├── distribution.py  # Distribution metrics (skewness, kurtosis, etc.)
│   ├── rolling.py       # Rolling window metrics
│   ├── validation.py    # Input validation utilities
│   ├── utils.py         # Helper functions
│   └── periods.py       # Period-based analysis
├── advanced/            # Advanced trading metrics
│   └── trades.py        # SQN, expectancy, smart ratios, etc.
├── institutional/       # Institutional-grade metrics
│   ├── robustness.py    # PSR, DSR, minimum track record
│   ├── volatility.py    # GARCH, ARCH tests
│   ├── var_extensions.py # Cornish-Fisher VaR, entropic VaR
│   ├── portfolio.py     # Ledoit-Wolf, correlation metrics
│   ├── systemic.py      # Absorption ratio, tail dependence
│   └── execution.py     # Market impact, implementation shortfall
├── trades/              # Trade conversion utilities
│   ├── conversion.py    # trades_to_returns
│   ├── equity.py        # Equity curve building
│   └── types.py         # Trade data types
└── reports/             # Report generation
    ├── metrics.py       # Metric calculations for reports
    └── html.py          # HTML report generation
```

## Current Branch
`phase-6-institutional-metrics` - Working on institutional-grade analytics
