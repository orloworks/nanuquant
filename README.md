# Polars Metrics

Native Polars metrics library for quantitative finance - a complete QuantStats replacement.

## Features

- **67 QuantStats metrics** fully replicated with native Polars
- **18 additional trading metrics** (SQN, Expectancy, K-ratio, etc.)
- **Zero pandas dependency** in production code path
- **Lazy evaluation support** for large datasets
- **Polars expression namespace** for idiomatic usage

## Installation

```bash
pip install polars-metrics
```

## Quick Start

```python
import polars as pl
import polars_metrics as pm

# Generate sample returns
returns = pl.Series("returns", [0.01, -0.02, 0.03, 0.01, -0.01, 0.02])

# Functional API
sharpe = pm.stats.sharpe(returns, rf=0.02)
sortino = pm.stats.sortino(returns)
max_dd = pm.stats.max_drawdown(returns)

# DataFrame with namespace (coming soon)
# import polars_metrics  # registers .metrics accessor
# df.select(pl.col("returns").metrics.sharpe())
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run differential tests against QuantStats
pytest tests/test_vs_quantstats.py -v

# Type checking
mypy polars_metrics
```

## License

MIT
