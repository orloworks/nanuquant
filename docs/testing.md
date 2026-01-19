# Testing Methodology

> **Disclaimer**: NanuQuant is for educational and research purposes only. Thorough testing does **NOT** guarantee accuracy for all use cases. Users should independently verify calculations for their specific applications. See [DISCLAIMER.md](../DISCLAIMER.md).

This document describes how NanuQuant is tested and validated.

---

## Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Test Suite Overview](#test-suite-overview)
- [Differential Testing vs QuantStats](#differential-testing-vs-quantstats)
- [Test Data and Fixtures](#test-data-and-fixtures)
- [Edge Case Testing](#edge-case-testing)
- [Integration Testing](#integration-testing)
- [Running Tests](#running-tests)
- [Known Differences](#known-differences)
- [Contributing Tests](#contributing-tests)

---

## Testing Philosophy

NanuQuant follows a multi-layered testing approach:

1. **Differential Testing**: Compare results against the established QuantStats library
2. **Mathematical Validation**: Verify formulas produce expected results on known inputs
3. **Edge Case Testing**: Test boundary conditions and error handling
4. **Integration Testing**: Validate with real market data
5. **Type Safety**: Strict mypy type checking

### Why Differential Testing?

Rather than just testing against expected values, we compare NanuQuant's output to QuantStats (a widely-used, battle-tested library). This approach:

- Validates we match industry-standard calculations
- Catches implementation bugs that might produce plausible-but-wrong results
- Documents intentional differences where we improve upon QuantStats

---

## Test Suite Overview

The test suite consists of 14 test modules with approximately 6,165 lines of test code.

| Module | Focus | Description |
|--------|-------|-------------|
| `test_vs_quantstats.py` | Differential validation | Core metrics compared to QuantStats |
| `test_timeseries.py` | Timeseries functions | Equity curves, drawdowns, cumulative returns |
| `test_trades.py` | Trade conversion | P&L from trade data |
| `test_new_metrics.py` | Advanced metrics | Trading metrics validation |
| `test_reports.py` | Report generation | HTML and metric aggregation |
| `test_institutional_robustness.py` | PSR/DSR | Robustness metrics |
| `test_institutional_volatility.py` | GARCH/ARCH | Volatility modeling |
| `test_institutional_var.py` | Advanced VaR | Cornish-Fisher, entropic VaR |
| `test_institutional_portfolio.py` | Portfolio analytics | MCR, Ledoit-Wolf |
| `test_institutional_execution.py` | Execution quality | Implementation shortfall |
| `test_institutional_systemic.py` | Systemic risk | Absorption ratio, tail dependence |
| `test_integration_market_data.py` | Real data validation | SPY, QQQ, BND fixtures |

---

## Differential Testing vs QuantStats

### Approach

We compare NanuQuant output against QuantStats for the same input data:

```python
import quantstats as qs
import nanuquant as nq
import numpy as np

def test_sharpe_matches_quantstats(sample_returns, polars_returns):
    # QuantStats result (pandas)
    qs_sharpe = qs.stats.sharpe(sample_returns)

    # NanuQuant result (polars)
    nq_sharpe = nq.sharpe(polars_returns)

    # Compare with tolerance
    assert np.isclose(nq_sharpe, qs_sharpe, rtol=TOLERANCE)
```

### Tolerance Levels

Different metrics require different tolerance levels:

| Tolerance | Value | Use Case |
|-----------|-------|----------|
| `EXACT` | 1e-10 | Deterministic calculations (sum, mean) |
| `TIGHT` | 1e-6 | Standard numerical operations |
| `LOOSE` | 1e-3 | Complex calculations with accumulated rounding |
| `STAT` | 0.05 | Statistical tests (p-values) |
| `CALENDAR_TOL` | 0.02 | Date-sensitive metrics (CAGR, Calmar) |
| `INTEGRATION_TOL` | 0.05 | Real market data tests |

### Metrics Tested Against QuantStats

**Core Metrics:**
- `sharpe` - Sharpe ratio
- `sortino` - Sortino ratio
- `max_drawdown` - Maximum drawdown
- `volatility` - Annualized volatility
- `cagr` - Compound annual growth rate
- `var` - Value at Risk
- `cvar` - Conditional VaR
- `calmar` - Calmar ratio
- `omega` - Omega ratio
- `win_rate` - Win rate
- `avg_win` / `avg_loss` - Average win/loss
- `profit_factor` - Profit factor
- `payoff_ratio` - Payoff ratio
- `skewness` / `kurtosis` - Distribution metrics
- And more...

---

## Test Data and Fixtures

### Synthetic Test Data

Located in `tests/conftest.py`:

```python
@pytest.fixture
def sample_returns():
    """1000 daily returns with injected patterns."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 1000)
    # Inject realistic patterns
    returns[100:150] *= 1.5  # Volatility cluster
    returns[500:510] = -0.03  # Drawdown event
    return pd.Series(returns)

@pytest.fixture
def polars_returns(sample_returns):
    """Polars version of sample returns."""
    return pl.Series(sample_returns.values)
```

### Real Market Data

Pre-cached parquet files eliminate network dependencies:

```python
@pytest.fixture
def spy_returns_full():
    """S&P 500 ETF returns (1993-present, ~8000+ observations)."""
    return pl.read_parquet("tests/data/spy_returns.parquet")

@pytest.fixture
def qqq_returns_full():
    """Nasdaq-100 ETF returns (1999-present, ~6700+ observations)."""
    return pl.read_parquet("tests/data/qqq_returns.parquet")

@pytest.fixture
def bnd_returns_full():
    """Bond ETF returns (2007-present, ~4700+ observations)."""
    return pl.read_parquet("tests/data/bnd_returns.parquet")
```

**Why cached data?**
- Reproducible tests (no API dependency)
- Faster test execution
- Consistent results across environments

---

## Edge Case Testing

### Boundary Conditions

We test edge cases that might cause errors:

```python
@pytest.fixture
def empty_returns():
    """Empty series."""
    return pl.Series([])

@pytest.fixture
def single_return():
    """Single observation."""
    return pl.Series([0.01])

@pytest.fixture
def all_positive_returns():
    """No losses (edge case for loss-based metrics)."""
    return pl.Series([0.01, 0.02, 0.015, 0.005, 0.03])

@pytest.fixture
def all_negative_returns():
    """No wins (edge case for win-based metrics)."""
    return pl.Series([-0.01, -0.02, -0.015, -0.005, -0.03])

@pytest.fixture
def flat_returns():
    """Zero returns."""
    return pl.Series([0.0, 0.0, 0.0, 0.0, 0.0])
```

### Expected Behaviors

| Edge Case | Expected Behavior |
|-----------|-------------------|
| Empty series | Raise `EmptySeriesError` |
| Single return | Return NaN or raise `InsufficientDataError` |
| All positive | `avg_loss` returns NaN, `profit_factor` returns inf |
| All negative | `avg_win` returns NaN, `win_rate` returns 0 |
| All zeros | `volatility` returns 0, `sharpe` returns NaN |
| Contains nulls | Nulls dropped automatically |

### Exception Testing

```python
def test_empty_series_raises():
    with pytest.raises(EmptySeriesError):
        nq.sharpe(pl.Series([]))

def test_insufficient_data_for_rolling():
    with pytest.raises(InsufficientDataError):
        nq.rolling_sharpe(pl.Series([0.01, 0.02]), window=100)
```

---

## Integration Testing

Integration tests use real market data to validate calculations in realistic conditions.

### Marking Integration Tests

```python
@pytest.mark.integration
def test_sharpe_with_real_spy_data(spy_returns_full):
    """Test Sharpe calculation on actual S&P 500 data."""
    sharpe = nq.sharpe(spy_returns_full)

    # Sanity check: S&P 500 historically has positive Sharpe
    assert 0 < sharpe < 2  # Reasonable range
```

### Multi-Asset Tests

```python
@pytest.mark.integration
def test_portfolio_metrics_with_market_data(market_data_df):
    """Test portfolio functions on SPY/QQQ/BND."""
    weights = [0.6, 0.3, 0.1]

    mcr = institutional.marginal_contribution_to_risk(market_data_df, weights)

    # PCR should sum to 1.0
    assert np.isclose(sum(mcr.pcr), 1.0)
```

### Running Integration Tests

```bash
# Include integration tests
pytest -m integration

# Exclude integration tests (faster)
pytest -m "not integration"
```

---

## Running Tests

### Basic Test Run

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Run specific test file
pytest tests/test_vs_quantstats.py

# Run specific test
pytest tests/test_vs_quantstats.py::test_sharpe_matches_quantstats
```

### Test Categories

```bash
# Differential tests only
pytest tests/test_vs_quantstats.py

# Institutional metrics tests
pytest tests/test_institutional_*.py

# Exclude slow tests
pytest -m "not benchmark"

# Include integration tests
pytest -m integration
```

### Coverage Report

```bash
# Install coverage
pip install pytest-cov

# Run with coverage
pytest --cov=nanuquant --cov-report=html

# View report
open htmlcov/index.html
```

### Type Checking

```bash
# Run mypy
mypy nanuquant

# Strict mode (default in pyproject.toml)
mypy nanuquant --strict
```

---

## Known Differences

NanuQuant intentionally differs from QuantStats in some areas:

| Metric | Difference | Rationale |
|--------|------------|-----------|
| **CAGR/Calmar** | Uses periods-based calculation | Works with any time series, not just datetime-indexed |
| **Treynor** | Uses CAGR/Beta | Standard academic definition |
| **Omega** | Fixed implementation | QuantStats has a bug in some versions |
| **CPC Index** | Uses `PF × WR × PR` | Standard trading literature formula |
| **Smart Sharpe/Sortino** | Uses Lo (2002) formula | Established academic reference |

These differences are documented and tests use appropriate tolerances.

---

## Contributing Tests

### Adding New Tests

1. **Follow existing patterns:**
```python
def test_new_metric_matches_expected(sample_returns, polars_returns):
    """Test new_metric calculation."""
    expected = calculate_expected_value(sample_returns)
    actual = nq.new_metric(polars_returns)

    assert np.isclose(actual, expected, rtol=TIGHT)
```

2. **Add edge case tests:**
```python
def test_new_metric_with_empty_series():
    with pytest.raises(EmptySeriesError):
        nq.new_metric(pl.Series([]))
```

3. **Add integration test if applicable:**
```python
@pytest.mark.integration
def test_new_metric_with_real_data(spy_returns_full):
    result = nq.new_metric(spy_returns_full)
    assert reasonable_bounds(result)
```

### Test Checklist

- [ ] Unit test with synthetic data
- [ ] Differential test vs QuantStats (if applicable)
- [ ] Edge case tests (empty, single, all positive, etc.)
- [ ] Error handling tests
- [ ] Type hints pass mypy
- [ ] Integration test with real data
- [ ] Documentation updated

---

## Continuous Integration

Tests run automatically on:
- Pull requests
- Merges to main branch
- Release tags

### CI Configuration

```yaml
# Example GitHub Actions workflow
test:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      python-version: ["3.10", "3.11", "3.12"]
  steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: pip install -e ".[dev]"
    - name: Run tests
      run: pytest -v
    - name: Type check
      run: mypy nanuquant
```

---

## See Also

- [API Reference](api/core.md)
- [Mathematical Formulas](mathematics.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
