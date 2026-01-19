# Institutional Metrics API Reference

> **Disclaimer**: NanuQuant is for educational and research purposes only. These institutional-grade metrics do **NOT** constitute financial advice. They require proper understanding and should be used alongside professional judgment. See [DISCLAIMER.md](../../DISCLAIMER.md).

This document covers institutional-grade metrics for professional quantitative analysis.

Module: `nanuquant.institutional`

---

## Table of Contents

- [Robustness Testing](#robustness-testing)
- [Volatility Modeling](#volatility-modeling)
- [Advanced Value at Risk](#advanced-value-at-risk)
- [Portfolio Analytics](#portfolio-analytics)
- [Systemic Risk](#systemic-risk)
- [Execution Quality](#execution-quality)

---

## Robustness Testing

Module: `nanuquant.institutional.robustness`

These metrics help determine if observed performance is statistically significant or potentially due to chance.

### probabilistic_sharpe_ratio

Calculates the probability that the true Sharpe ratio exceeds a benchmark value.

```python
from nanuquant import institutional

psr = institutional.probabilistic_sharpe_ratio(
    returns: pl.Series,
    benchmark_sr: float = 0.0,
    periods_per_year: int = 252
) -> PSRResult
```

**Parameters:**
- `returns`: Series of period returns
- `benchmark_sr`: Benchmark Sharpe ratio to compare against (default 0)
- `periods_per_year`: Annualization factor

**Returns:** `PSRResult` NamedTuple containing:
- `psr`: Probabilistic Sharpe Ratio (probability 0-1)
- `observed_sr`: The observed Sharpe ratio
- `sr_std_error`: Standard error of the Sharpe estimate
- `skewness`: Return distribution skewness
- `kurtosis`: Return distribution excess kurtosis

**Example:**
```python
result = institutional.probabilistic_sharpe_ratio(returns, benchmark_sr=0.5)
print(f"Probability SR > 0.5: {result.psr:.2%}")
print(f"Observed SR: {result.observed_sr:.4f}")
```

**Interpretation:**
- PSR > 0.95: High confidence that true SR exceeds benchmark
- PSR < 0.50: True SR likely below benchmark
- Accounts for non-normality (skewness and kurtosis)

**Reference:** Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."

---

### deflated_sharpe_ratio

Adjusts for multiple testing (strategy selection bias).

```python
dsr = institutional.deflated_sharpe_ratio(
    returns: pl.Series,
    n_trials: int,
    benchmark_sr: float = 0.0,
    periods_per_year: int = 252
) -> float
```

**Parameters:**
- `returns`: Series of period returns
- `n_trials`: Number of strategy variations tested
- `benchmark_sr`: Benchmark Sharpe ratio
- `periods_per_year`: Annualization factor

**Returns:** Deflated Sharpe Ratio (probability 0-1)

**Example:**
```python
# If you tested 100 parameter combinations
dsr = institutional.deflated_sharpe_ratio(returns, n_trials=100)
print(f"Deflated SR: {dsr:.4f}")
```

**Interpretation:**
- DSR penalizes for the number of strategies tested
- DSR > 0.95: Strategy likely robust despite multiple testing
- DSR near 0: Observed performance likely due to selection bias

**Warning:** This is critical when backtesting multiple strategy variations. A Sharpe of 2.0 found after testing 1000 variations may be pure luck.

**Reference:** Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio."

---

### minimum_track_record_length

Minimum number of periods needed for statistical significance.

```python
min_trl = institutional.minimum_track_record_length(
    observed_sr: float,
    benchmark_sr: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95
) -> int
```

**Parameters:**
- `observed_sr`: Observed annualized Sharpe ratio
- `benchmark_sr`: Benchmark Sharpe ratio
- `skewness`: Return skewness
- `kurtosis`: Return kurtosis
- `confidence`: Required confidence level

**Returns:** Minimum number of periods required

**Example:**
```python
# How long a track record do we need?
min_periods = institutional.minimum_track_record_length(
    observed_sr=1.5,
    benchmark_sr=0.0,
    confidence=0.95
)
print(f"Minimum track record: {min_periods} periods")
```

---

## Volatility Modeling

Module: `nanuquant.institutional.volatility`

### arch_effect_test

Engle's Lagrange Multiplier test for ARCH effects (volatility clustering).

```python
result = institutional.arch_effect_test(
    returns: pl.Series,
    lags: int = 12
) -> ARCHTestResult
```

**Parameters:**
- `returns`: Series of period returns
- `lags`: Number of lags to test

**Returns:** `ARCHTestResult` NamedTuple containing:
- `statistic`: LM test statistic
- `p_value`: P-value of the test
- `lags`: Number of lags used
- `has_arch_effects`: Boolean (True if p_value < 0.05)

**Example:**
```python
result = institutional.arch_effect_test(returns)
if result.has_arch_effects:
    print("Volatility clustering detected - consider GARCH modeling")
```

**Use Case:** Determines if returns exhibit time-varying volatility, which would make constant-volatility assumptions invalid.

---

### garch_volatility

Fit GARCH(1,1) model for conditional volatility estimation.

```python
result = institutional.garch_volatility(
    returns: pl.Series,
    periods_per_year: int = 252
) -> GARCHResult
```

**Parameters:**
- `returns`: Series of period returns
- `periods_per_year`: Annualization factor

**Returns:** `GARCHResult` NamedTuple containing:
- `omega`: Constant term (long-run variance weight)
- `alpha`: ARCH coefficient (shock impact)
- `beta`: GARCH coefficient (persistence)
- `conditional_volatility`: Series of conditional volatilities
- `persistence`: alpha + beta (should be < 1 for stationarity)
- `long_run_variance`: omega / (1 - persistence)
- `forecast`: One-step-ahead volatility forecast

**Example:**
```python
garch = institutional.garch_volatility(returns)
print(f"Persistence: {garch.persistence:.4f}")
print(f"Current Vol Forecast: {garch.forecast:.2%}")

# Plot conditional volatility
import matplotlib.pyplot as plt
plt.plot(garch.conditional_volatility)
plt.title("GARCH(1,1) Conditional Volatility")
```

**Interpretation:**
- High persistence (> 0.95): Volatility shocks are long-lasting
- High alpha: Recent shocks have large impact
- High beta: Volatility changes slowly

**Warning:** GARCH models require sufficient data (typically 500+ observations) for reliable estimation.

---

## Advanced Value at Risk

Module: `nanuquant.institutional.var_extensions`

### parametric_var

Standard parametric VaR assuming normality.

```python
var = institutional.parametric_var(
    returns: pl.Series,
    confidence: float = 0.95
) -> float
```

---

### historical_var

Non-parametric VaR using historical quantiles.

```python
var = institutional.historical_var(
    returns: pl.Series,
    confidence: float = 0.95
) -> float
```

**Advantage:** No distributional assumptions
**Disadvantage:** Requires sufficient historical data

---

### cornish_fisher_var

VaR adjusted for skewness and kurtosis using Cornish-Fisher expansion.

```python
cf_var = institutional.cornish_fisher_var(
    returns: pl.Series,
    confidence: float = 0.95
) -> float
```

**Returns:** Cornish-Fisher adjusted VaR

**Advantage:** Accounts for non-normal distribution features
**Use Case:** More accurate VaR when returns exhibit fat tails or skewness

---

### modified_var

Modified VaR with full higher-moment adjustments.

```python
m_var = institutional.modified_var(
    returns: pl.Series,
    confidence: float = 0.95
) -> float
```

---

### entropic_var

Entropic VaR - a coherent risk measure based on exponential utility.

```python
e_var = institutional.entropic_var(
    returns: pl.Series,
    confidence: float = 0.95
) -> float
```

**Advantage:** Coherent risk measure (satisfies subadditivity)
**Use Case:** Portfolio optimization where VaR's non-coherence causes issues

---

## Portfolio Analytics

Module: `nanuquant.institutional.portfolio`

### marginal_contribution_to_risk

Decompose portfolio risk by asset contribution.

```python
result = institutional.marginal_contribution_to_risk(
    returns: pl.DataFrame,
    weights: list[float]
) -> MCRResult
```

**Parameters:**
- `returns`: DataFrame with asset returns as columns
- `weights`: Portfolio weights (must sum to 1)

**Returns:** `MCRResult` NamedTuple containing:
- `mcr`: Marginal contribution to risk per asset
- `pcr`: Percentage contribution to risk per asset
- `portfolio_volatility`: Total portfolio volatility

**Example:**
```python
returns_df = pl.DataFrame({
    "stock_a": [...],
    "stock_b": [...],
    "stock_c": [...]
})
weights = [0.4, 0.35, 0.25]

result = institutional.marginal_contribution_to_risk(returns_df, weights)
for asset, pcr in zip(returns_df.columns, result.pcr):
    print(f"{asset}: {pcr:.1%} of total risk")
```

**Use Case:** Identify which assets contribute most to portfolio risk for potential rebalancing.

---

### ledoit_wolf_covariance

Shrinkage estimator for more stable covariance matrix.

```python
result = institutional.ledoit_wolf_covariance(
    returns: pl.DataFrame
) -> LedoitWolfResult
```

**Returns:** `LedoitWolfResult` NamedTuple containing:
- `covariance`: Shrunk covariance matrix
- `shrinkage`: Optimal shrinkage intensity (0-1)

**Advantage:** More stable than sample covariance, especially with limited data
**Use Case:** Portfolio optimization with many assets or limited history

**Reference:** Ledoit, O., & Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices."

---

### portfolio_volatility

Calculate portfolio volatility from asset returns and weights.

```python
vol = institutional.portfolio_volatility(
    returns: pl.DataFrame,
    weights: list[float],
    periods_per_year: int = 252
) -> float
```

---

### correlation_from_covariance

Extract correlation matrix from covariance matrix.

```python
corr = institutional.correlation_from_covariance(
    covariance: np.ndarray
) -> np.ndarray
```

---

## Systemic Risk

Module: `nanuquant.institutional.systemic`

### absorption_ratio

Measures systemic fragility based on PCA.

```python
ar = institutional.absorption_ratio(
    returns: pl.DataFrame,
    n_components: int = 5
) -> float
```

**Parameters:**
- `returns`: DataFrame with asset returns as columns
- `n_components`: Number of principal components

**Returns:** Fraction of variance explained by top components (0-1)

**Interpretation:**
- High AR (> 0.8): Market is "tight" - assets move together, fragile
- Low AR (< 0.5): Market is "loose" - diverse, more stable

**Use Case:** Identify periods of elevated systemic risk when correlations increase.

**Reference:** Kritzman, M., et al. (2011). "Principal Components as a Measure of Systemic Risk."

---

### lower_tail_dependence

Measures co-movement in extreme down markets.

```python
ltd = institutional.lower_tail_dependence(
    returns_a: pl.Series,
    returns_b: pl.Series,
    threshold: float = 0.05
) -> float
```

**Parameters:**
- `returns_a`, `returns_b`: Two return series
- `threshold`: Percentile threshold for "extreme" (default 5%)

**Returns:** Tail dependence coefficient (0-1)

**Interpretation:**
- 0: No tail dependence (diversification works in crises)
- 1: Perfect tail dependence (crash together)

**Use Case:** Assess diversification benefits during market stress.

---

### downside_correlation

Correlation computed only during down markets.

```python
down_corr = institutional.downside_correlation(
    returns_a: pl.Series,
    returns_b: pl.Series,
    threshold: float = 0.0
) -> float
```

---

### upside_correlation

Correlation computed only during up markets.

```python
up_corr = institutional.upside_correlation(
    returns_a: pl.Series,
    returns_b: pl.Series,
    threshold: float = 0.0
) -> float
```

**Use Case:** Identify asymmetric correlations - assets that are more correlated in down markets than up markets provide less diversification when needed most.

---

## Execution Quality

Module: `nanuquant.institutional.execution`

### implementation_shortfall

Measures the gap between paper performance and actual execution.

```python
result = institutional.implementation_shortfall(
    decision_price: float,
    execution_price: float,
    benchmark_price: float,
    quantity: float,
    direction: str  # "buy" or "sell"
) -> ImplementationShortfallResult
```

**Returns:** `ImplementationShortfallResult` NamedTuple containing:
- `total_is`: Total implementation shortfall
- `delay_cost`: Cost from decision to execution start
- `market_impact`: Estimated market impact
- `timing_cost`: Residual timing costs

**Use Case:** Analyze trading costs and execution quality.

---

### market_impact_estimate

Estimate market impact using square-root model.

```python
impact = institutional.market_impact_estimate(
    volume: float,
    adv: float,  # Average Daily Volume
    volatility: float,
    participation_rate: float = 0.1
) -> float
```

**Formula:** Impact = k × σ × √(V / ADV)

**Use Case:** Pre-trade estimation of execution costs.

---

### vwap_slippage

Calculate slippage versus Volume-Weighted Average Price.

```python
slippage = institutional.vwap_slippage(
    execution_price: float,
    vwap: float,
    direction: str  # "buy" or "sell"
) -> float
```

**Returns:** Slippage as percentage (positive = worse than VWAP)

---

### spread_cost

Calculate the cost of bid-ask spread.

```python
cost = institutional.spread_cost(
    bid: float,
    ask: float,
    quantity: float,
    direction: str
) -> float
```

---

### execution_vwap

Calculate VWAP from execution data.

```python
vwap = institutional.execution_vwap(
    prices: pl.Series,
    volumes: pl.Series
) -> float
```

---

## Usage Examples

### Complete Robustness Check

```python
from nanuquant import institutional
import polars as pl

returns = pl.Series([...])  # Your strategy returns

# 1. Check for statistical significance
psr = institutional.probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
print(f"PSR (SR > 0): {psr.psr:.2%}")

# 2. Adjust for multiple testing
dsr = institutional.deflated_sharpe_ratio(returns, n_trials=50)
print(f"DSR (after 50 trials): {dsr:.2%}")

# 3. Check required track record
min_trl = institutional.minimum_track_record_length(
    observed_sr=psr.observed_sr,
    skewness=psr.skewness,
    kurtosis=psr.kurtosis
)
print(f"Minimum track record needed: {min_trl} periods")
print(f"Current track record: {len(returns)} periods")

# 4. Test for volatility clustering
arch = institutional.arch_effect_test(returns)
if arch.has_arch_effects:
    print("Warning: Returns exhibit volatility clustering")
    garch = institutional.garch_volatility(returns)
    print(f"GARCH persistence: {garch.persistence:.4f}")
```

### Portfolio Risk Decomposition

```python
from nanuquant import institutional
import polars as pl

# Multi-asset portfolio
returns_df = pl.DataFrame({
    "SPY": [...],
    "AGG": [...],
    "GLD": [...]
})
weights = [0.6, 0.3, 0.1]

# Get risk contributions
mcr = institutional.marginal_contribution_to_risk(returns_df, weights)
print("\nRisk Contributions:")
for asset, pct in zip(returns_df.columns, mcr.pcr):
    print(f"  {asset}: {pct:.1%}")

# Use Ledoit-Wolf for stable covariance
lw = institutional.ledoit_wolf_covariance(returns_df)
print(f"\nShrinkage intensity: {lw.shrinkage:.4f}")

# Check systemic risk
ar = institutional.absorption_ratio(returns_df)
print(f"Absorption Ratio: {ar:.2%}")
```

---

## See Also

- [Core Metrics](core.md)
- [Advanced Trading Metrics](advanced.md)
- [Mathematical Formulas](../mathematics.md)
- [Testing Methodology](../testing.md)
