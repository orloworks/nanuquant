# Contributing to NanuQuant

Thank you for your interest in contributing to NanuQuant! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
git clone https://github.com/launchstack-dev/nanuquant.git
cd nanuquant
uv pip install -e ".[dev]"
```

## Development Workflow

### Running Tests

```bash
# All tests
pytest

# Differential tests against QuantStats
pytest tests/test_vs_quantstats.py -v

# Integration tests with real market data
pytest -m integration

# Exclude benchmarks
pytest -m "not benchmark"
```

### Code Quality

NanuQuant enforces strict code quality. Run all checks before submitting a PR:

```bash
# Linting
ruff check nanuquant

# Formatting
ruff format --check nanuquant

# Type checking (strict mode)
mypy nanuquant
```

### Running Benchmarks

```bash
python benchmarks/run_benchmarks.py --output docs/benchmarks.md
```

## Making Changes

### Code Style

- **Line length**: 100 characters
- **Type hints**: Required on all public functions (strict mypy)
- **Docstrings**: Google style
- **Imports**: Sorted by ruff (isort-compatible)

### Adding a New Metric

1. Implement in the appropriate module (`core/`, `advanced/`, or `institutional/`)
2. Export from `__init__.py`
3. Add to the Polars `.metrics` namespace in `namespace.py` (if appropriate)
4. Write tests:
   - Unit test with synthetic data
   - Differential test vs QuantStats (if comparable metric exists)
   - Edge case tests (empty, single observation, all positive, all negative, flat)
   - Integration test with real market data
5. Add documentation to the relevant API doc in `docs/api/`
6. Add the mathematical formula to `docs/mathematics.md`

### Test Data

- **Synthetic data** is generated in `tests/conftest.py` with `seed=42` for reproducibility
- **Real market data** (SPY, QQQ, BND) is pre-cached in `tests/.data_cache/` — no network required
- Never add tests that require network access at runtime

### Tolerance Levels

Use the appropriate tolerance when comparing numerical results:

| Tolerance | Value | Use For |
|-----------|-------|---------|
| EXACT | 1e-10 | Deterministic operations (sum, mean, min, max) |
| TIGHT | 1e-8 | Standard numerical operations |
| LOOSE | 1e-6 | Complex accumulated rounding |
| STAT | 1e-4 | Statistical tests |
| CALENDAR | 0.003 | Calendar-sensitive metrics (CAGR, Calmar) |

## Pull Request Guidelines

1. Create a feature branch from `main`
2. Keep PRs focused — one feature or fix per PR
3. All tests must pass
4. All linting/typing checks must pass
5. Add tests for new functionality
6. Update documentation if adding/changing public API

## Architecture Notes

- **Zero Pandas in production** — Pandas is only used in tests (for QuantStats comparison)
- All public functions accept `polars.Series` and return Python scalars or `polars.Series`
- Use Polars expressions internally, not NumPy loops (except where SciPy is needed)
- Custom exceptions in `nanuquant/exceptions.py` — always raise the appropriate one

## Questions?

Open an issue on [GitHub](https://github.com/launchstack-dev/nanuquant/issues).
