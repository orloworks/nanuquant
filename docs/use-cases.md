# Use Cases and Practical Examples

> **IMPORTANT DISCLAIMER**: NanuQuant is for **educational and research purposes only**. The examples in this document do **NOT** constitute financial advice, investment recommendations, or trading signals. Past performance does not guarantee future results. Always consult a qualified financial professional before making investment decisions. See [DISCLAIMER](https://github.com/launchstack-dev/nanuquant/blob/main/DISCLAIMER.md).

This document provides practical use cases for NanuQuant metrics, along with important considerations for each.

---

## Table of Contents

- [Strategy Performance Evaluation](#strategy-performance-evaluation)
- [Risk Assessment](#risk-assessment)
- [Strategy Comparison](#strategy-comparison)
- [Robustness Testing](#robustness-testing)
- [Portfolio Analysis](#portfolio-analysis)
- [Execution Quality Analysis](#execution-quality-analysis)
- [Distribution Analysis](#distribution-analysis)
- [Educational and Research Applications](#educational-and-research-applications)

---

## Strategy Performance Evaluation

### Use Case: Backtesting Analysis

Evaluate the historical performance of a trading strategy.

```python
import polars as pl
import nanuquant as nq
from nanuquant.reports import full_metrics

# Strategy returns from backtest (NOT real trading results)
returns = pl.Series("returns", [...])  # Your backtest returns

# Comprehensive analysis
report = full_metrics(returns)

print("=== Performance Summary ===")
print(f"CAGR: {report.returns['cagr']:.2%}")
print(f"Volatility: {report.risk['volatility']:.2%}")
print(f"Sharpe Ratio: {report.performance['sharpe']:.2f}")
print(f"Sortino Ratio: {report.performance['sortino']:.2f}")
print(f"Max Drawdown: {report.risk['max_drawdown']:.2%}")
print(f"Calmar Ratio: {report.performance['calmar']:.2f}")
```

**Relevant Metrics:**
| Metric | Purpose | Limitation |
|--------|---------|------------|
| CAGR | Annualized return | Doesn't reflect risk; past returns don't predict future |
| Sharpe | Risk-adjusted return | Assumes normal returns; penalizes upside volatility |
| Sortino | Downside risk-adjusted return | Better for asymmetric returns; still backward-looking |
| Max Drawdown | Worst loss from peak | Historical max may be exceeded in future |
| Calmar | Return per drawdown risk | Sensitive to observation period |

> **Warning**: Backtested results are NOT indicative of future performance. Backtests suffer from survivorship bias, look-ahead bias, overfitting, and do not account for real-world execution costs.

---

### Use Case: Trading System Quality

Assess the quality of a systematic trading approach.

```python
# Trading system metrics
sqn = nq.sqn(returns)
expectancy = nq.expectancy(returns)
k_ratio = nq.k_ratio(returns)
cpc = nq.cpc_index(returns)

print("=== System Quality ===")
print(f"SQN: {sqn:.2f}")
print(f"Expectancy: {expectancy:.4f}")
print(f"K-Ratio: {k_ratio:.2f}")
print(f"CPC Index: {cpc:.2f}")

# Interpret SQN
if sqn >= 3.0:
    print("SQN indicates excellent system (but verify with out-of-sample)")
elif sqn >= 2.0:
    print("SQN indicates average system")
else:
    print("SQN indicates below-average system")
```

**Relevant Metrics:**
| Metric | Purpose | Interpretation |
|--------|---------|----------------|
| SQN | System quality number | > 2.5 good, > 3.0 excellent (be skeptical of > 7) |
| Expectancy | Expected value per trade | Must be positive for viable system |
| K-Ratio | Equity curve consistency | Higher = more consistent growth |
| CPC Index | Combined quality measure | > 1.2 strong, < 0.8 weak |

> **Warning**: High system quality metrics on backtests often indicate overfitting. Always validate on out-of-sample data.

---

## Risk Assessment

### Use Case: Portfolio Risk Monitoring

Monitor and understand portfolio risk characteristics.

```python
# Risk metrics
vol = nq.volatility(returns)
max_dd = nq.max_drawdown(returns)
var_95 = nq.var(returns, confidence=0.95)
cvar_95 = nq.cvar(returns, confidence=0.95)

print("=== Risk Profile ===")
print(f"Annualized Volatility: {vol:.2%}")
print(f"Maximum Drawdown: {max_dd:.2%}")
print(f"95% VaR (daily): {var_95:.2%}")
print(f"95% CVaR (daily): {cvar_95:.2%}")

# Advanced VaR for non-normal returns
from nanuquant import institutional

cf_var = institutional.cornish_fisher_var(returns, confidence=0.95)
print(f"95% Cornish-Fisher VaR: {cf_var:.2%}")
```

**Relevant Metrics:**
| Metric | Purpose | Limitation |
|--------|---------|------------|
| Volatility | Overall risk measure | Symmetric; doesn't distinguish up vs down |
| VaR | Potential loss at confidence level | Assumes distribution; not coherent measure |
| CVaR | Expected loss beyond VaR | More conservative but still model-dependent |
| Max Drawdown | Worst historical loss | Future drawdowns may exceed historical |

> **Warning**: VaR and related metrics are model-dependent. Parametric VaR assumes normality; real markets exhibit fat tails. These metrics should be one input among many in risk management.

---

### Use Case: Downside Risk Analysis

Focus specifically on loss scenarios.

```python
# Downside-focused metrics
downside_dev = nq.downside_deviation(returns)
ulcer = nq.ulcer_index(returns)
sortino = nq.sortino(returns)
tail = nq.tail_ratio(returns)

print("=== Downside Analysis ===")
print(f"Downside Deviation: {downside_dev:.2%}")
print(f"Ulcer Index: {ulcer:.4f}")
print(f"Sortino Ratio: {sortino:.2f}")
print(f"Tail Ratio: {tail:.2f}")

# Distribution of losses
skew = nq.skewness(returns)
kurt = nq.kurtosis(returns)
print(f"\nSkewness: {skew:.2f} {'(negative = left tail risk)' if skew < 0 else ''}")
print(f"Excess Kurtosis: {kurt:.2f} {'(fat tails)' if kurt > 0 else ''}")
```

**Relevant Metrics:**
| Metric | Purpose | Interpretation |
|--------|---------|----------------|
| Downside Deviation | Risk of losses only | Used in Sortino; focuses on negative returns |
| Ulcer Index | Drawdown severity | Combines depth and duration of drawdowns |
| Tail Ratio | Gain magnitude vs loss magnitude | > 1 means bigger wins than losses |
| Skewness | Distribution asymmetry | Negative = larger left tail (more crash risk) |

> **Warning**: Historical downside metrics may underestimate future risk. Markets can experience unprecedented events ("black swans").

---

## Strategy Comparison

### Use Case: Comparing Multiple Strategies

Compare different strategies or parameter variations.

```python
import polars as pl
import nanuquant as nq

# Multiple strategy returns
strategy_a = pl.Series([...])
strategy_b = pl.Series([...])
benchmark = pl.Series([...])

def compare_strategies(returns_dict, benchmark):
    results = {}
    for name, returns in returns_dict.items():
        results[name] = {
            "cagr": nq.cagr(returns),
            "volatility": nq.volatility(returns),
            "sharpe": nq.sharpe(returns),
            "sortino": nq.sortino(returns),
            "max_dd": nq.max_drawdown(returns),
            "info_ratio": nq.information_ratio(returns, benchmark),
        }
    return pl.DataFrame(results).transpose()

comparison = compare_strategies(
    {"Strategy A": strategy_a, "Strategy B": strategy_b},
    benchmark
)
print(comparison)
```

**Comparison Considerations:**

| Factor | Question to Ask |
|--------|-----------------|
| Time Period | Are strategies compared over the same period? |
| Market Regime | Does one strategy favor certain conditions? |
| Transaction Costs | Are costs consistently applied? |
| Sample Size | Is there enough data for statistical significance? |
| Risk Profile | Is higher return due to higher risk? |

> **Warning**: Strategy comparison on backtests is prone to selection bias. The "best" backtest often performs worst in live trading due to overfitting.

---

### Use Case: Benchmark-Relative Analysis

Evaluate performance relative to a benchmark.

```python
# Benchmark comparison
alpha, beta = nq.greeks(returns, benchmark)
info_ratio = nq.information_ratio(returns, benchmark)
r_squared = nq.r_squared(returns, benchmark)
treynor = nq.treynor_ratio(returns, benchmark)
correlation = nq.benchmark_correlation(returns, benchmark)

print("=== Benchmark Analysis ===")
print(f"Alpha (annualized): {alpha:.2%}")
print(f"Beta: {beta:.2f}")
print(f"Information Ratio: {info_ratio:.2f}")
print(f"R-Squared: {r_squared:.2%}")
print(f"Treynor Ratio: {treynor:.2f}")
print(f"Correlation: {correlation:.2f}")
```

**Relevant Metrics:**
| Metric | Purpose | Interpretation |
|--------|---------|----------------|
| Alpha | Excess return after beta adjustment | Positive = outperformance (if statistically significant) |
| Beta | Systematic risk exposure | 1.0 = market risk; > 1 = amplified |
| Information Ratio | Active return per tracking error | > 0.5 good, > 1.0 excellent |
| R-Squared | Explained by benchmark movements | High = benchmark-driven returns |

> **Warning**: Alpha is backward-looking and often not persistent. Historical alpha does not guarantee future outperformance.

---

## Robustness Testing

### Use Case: Statistical Significance Check

Determine if observed performance is statistically meaningful.

```python
from nanuquant import institutional

# Is the Sharpe ratio statistically significant?
psr_result = institutional.probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)

print("=== Statistical Significance ===")
print(f"Observed Sharpe: {psr_result.observed_sr:.2f}")
print(f"Probability SR > 0: {psr_result.psr:.2%}")
print(f"Skewness: {psr_result.skewness:.2f}")
print(f"Kurtosis: {psr_result.kurtosis:.2f}")

if psr_result.psr > 0.95:
    print("High confidence that true Sharpe exceeds 0")
else:
    print("Cannot confidently conclude positive Sharpe")
```

**Relevant Metrics:**
| Metric | Purpose | Threshold |
|--------|---------|-----------|
| PSR | Probability true SR > benchmark | > 95% for high confidence |
| Skewness | Non-normality adjustment | Negative reduces PSR |
| Kurtosis | Fat tail adjustment | High kurtosis reduces PSR |

> **Warning**: Even statistically significant results can fail out-of-sample due to regime changes.

---

### Use Case: Multiple Testing Adjustment

Account for testing multiple strategy variations.

```python
from nanuquant import institutional

# If you tested 100 parameter combinations
n_variations_tested = 100

dsr = institutional.deflated_sharpe_ratio(
    returns,
    n_trials=n_variations_tested,
    benchmark_sr=0.0
)

min_trl = institutional.minimum_track_record_length(
    observed_sr=nq.sharpe(returns),
    benchmark_sr=0.0
)

print(f"=== Multiple Testing Adjustment ===")
print(f"Strategies tested: {n_variations_tested}")
print(f"Deflated Sharpe Probability: {dsr:.2%}")
print(f"Minimum track record needed: {min_trl} periods")
print(f"Current track record: {len(returns)} periods")

if dsr > 0.95:
    print("Strategy remains significant after multiple testing adjustment")
else:
    print("Warning: Performance may be due to selection bias")
```

> **Critical Warning**: Backtest optimization with many trials almost guarantees finding a "profitable" strategy by chance. DSR helps quantify this risk but cannot eliminate it.

---

## Portfolio Analysis

### Use Case: Risk Contribution Analysis

Understand which assets drive portfolio risk.

```python
from nanuquant import institutional
import polars as pl

# Multi-asset portfolio
returns_df = pl.DataFrame({
    "stocks": [...],
    "bonds": [...],
    "commodities": [...]
})
weights = [0.6, 0.3, 0.1]

# Risk decomposition
mcr_result = institutional.marginal_contribution_to_risk(returns_df, weights)

print("=== Risk Contribution Analysis ===")
print(f"Portfolio Volatility: {mcr_result.portfolio_volatility:.2%}")
print("\nPercentage Contribution to Risk:")
for asset, pcr in zip(returns_df.columns, mcr_result.pcr):
    print(f"  {asset}: {pcr:.1%}")
```

**Relevant Metrics:**
| Metric | Purpose | Use |
|--------|---------|-----|
| MCR | Marginal contribution to risk | How much risk changes with weight change |
| PCR | Percentage contribution to risk | Which assets dominate portfolio risk |

> **Warning**: Risk contributions are based on historical correlations, which can change dramatically in crisis periods.

---

### Use Case: Correlation Analysis

Assess diversification benefits.

```python
from nanuquant import institutional

# Pairwise correlation analysis
stocks = pl.Series([...])
bonds = pl.Series([...])

# Regular correlation
corr = nq.benchmark_correlation(stocks, bonds)

# Asymmetric correlation
down_corr = institutional.downside_correlation(stocks, bonds)
up_corr = institutional.upside_correlation(stocks, bonds)
tail_dep = institutional.lower_tail_dependence(stocks, bonds)

print("=== Correlation Analysis ===")
print(f"Overall Correlation: {corr:.2f}")
print(f"Upside Correlation: {up_corr:.2f}")
print(f"Downside Correlation: {down_corr:.2f}")
print(f"Tail Dependence: {tail_dep:.2f}")

if down_corr > up_corr:
    print("\nWarning: Higher correlation in down markets - diversification weakens when needed most")
```

**Relevant Metrics:**
| Metric | Purpose | Implication |
|--------|---------|-------------|
| Downside Correlation | Correlation in losses | High = less diversification in crashes |
| Tail Dependence | Extreme co-movement | High = joint crashes likely |

> **Warning**: Correlations increase during market stress. Diversification benefits often fail precisely when needed most.

---

## Execution Quality Analysis

### Use Case: Trading Cost Analysis

Analyze execution quality and trading costs.

```python
from nanuquant import institutional

# Execution analysis for a multi-tranche order
is_result = institutional.implementation_shortfall(
    decision_price=100.00,
    execution_prices=[100.25, 100.30],  # Two fills
    quantities=[500, 500],
    side=1,  # 1 for buy, -1 for sell
    arrival_price=100.10,
    end_price=101.00
)

print("=== Execution Analysis ===")
print(f"Total Implementation Shortfall: ${is_result.total_shortfall:.2f}")
print(f"Shortfall: {is_result.shortfall_bps:.2f} bps")
print(f"Delay Cost: ${is_result.delay_cost:.2f}")
print(f"Trading Cost: ${is_result.trading_cost:.2f}")

# Pre-trade impact estimate
impact_est = institutional.market_impact_estimate(
    trade_volume=10000,
    avg_daily_volume=1000000,  # 1M average daily volume
    volatility=0.20,  # 20% annualized vol
    impact_coefficient=0.1
)
print(f"\nEstimated Market Impact: {impact_est:.4%}")
```

**Relevant Metrics:**
| Metric | Purpose | Note |
|--------|---------|------|
| Implementation Shortfall | Total execution cost | Benchmark for execution quality |
| Market Impact | Price movement from trading | Square-root model is approximation |
| VWAP Slippage | Deviation from VWAP | Common execution benchmark |

> **Warning**: Market impact models are approximations. Actual impact varies with market conditions, security liquidity, and execution timing.

---

## Distribution Analysis

### Use Case: Return Distribution Analysis

Understand the statistical properties of returns.

```python
# Distribution metrics
skew = nq.skewness(returns)
kurt = nq.kurtosis(returns)
jb_stat, jb_pval = nq.jarque_bera(returns)

print("=== Distribution Analysis ===")
print(f"Skewness: {skew:.3f}")
print(f"Excess Kurtosis: {kurt:.3f}")
print(f"Jarque-Bera p-value: {jb_pval:.4f}")

if jb_pval < 0.05:
    print("\nReturns are NOT normally distributed")
    print("Consider using Cornish-Fisher VaR or Smart Sharpe")
else:
    print("\nCannot reject normality")
```

**Interpretation Guide:**
| Statistic | Value | Meaning |
|-----------|-------|---------|
| Skewness < 0 | Negative | Left tail risk (large losses more likely than large gains) |
| Kurtosis > 0 | Positive | Fat tails (extreme events more likely than normal) |
| JB p-value < 0.05 | Significant | Returns are non-normal |

> **Warning**: Non-normal returns invalidate many standard assumptions. Use robust metrics when normality is rejected.

---

## Educational and Research Applications

### Use Case: Teaching Finance Concepts

NanuQuant is well-suited for educational purposes:

```python
import polars as pl
import nanuquant as nq

# Demonstrate Sharpe ratio concept
def explain_sharpe():
    """Educational example of Sharpe ratio."""

    # Two strategies with same return, different risk
    conservative = pl.Series([0.005, 0.006, 0.004, 0.005, 0.005] * 50)
    aggressive = pl.Series([0.02, -0.01, 0.03, -0.02, 0.01] * 50)

    print("=== Sharpe Ratio Demonstration ===")
    print("\nConservative Strategy:")
    print(f"  Mean Return: {conservative.mean():.4f}")
    print(f"  Volatility: {nq.volatility(conservative):.4f}")
    print(f"  Sharpe: {nq.sharpe(conservative):.2f}")

    print("\nAggressive Strategy:")
    print(f"  Mean Return: {aggressive.mean():.4f}")
    print(f"  Volatility: {nq.volatility(aggressive):.4f}")
    print(f"  Sharpe: {nq.sharpe(aggressive):.2f}")

    print("\nConclusion: Higher Sharpe = better risk-adjusted return")
    print("(But remember: This is educational, not financial advice!)")
```

### Use Case: Academic Research

```python
# Research application: Testing market efficiency hypothesis
def analyze_autocorrelation_effect():
    """Research example: Impact of autocorrelation on Sharpe."""

    returns = pl.Series([...])  # Research data

    standard_sharpe = nq.sharpe(returns)
    smart_sharpe = nq.smart_sharpe(returns)

    adjustment_factor = smart_sharpe / standard_sharpe if standard_sharpe != 0 else 0

    print(f"Standard Sharpe: {standard_sharpe:.4f}")
    print(f"Smart Sharpe: {smart_sharpe:.4f}")
    print(f"Autocorrelation Adjustment: {adjustment_factor:.2%}")

    # Reference: Lo (2002) "The Statistics of Sharpe Ratios"
```

---

## Summary: Choosing the Right Metrics

| Goal | Primary Metrics | Secondary Metrics |
|------|-----------------|-------------------|
| **Performance** | CAGR, Sharpe, Sortino | Calmar, Omega, Information Ratio |
| **Risk** | Volatility, Max Drawdown, CVaR | Ulcer Index, Cornish-Fisher VaR |
| **Quality** | SQN, Expectancy, K-Ratio | CPC Index, Profit Factor |
| **Robustness** | PSR, DSR | Min Track Record Length |
| **Distribution** | Skewness, Kurtosis | Jarque-Bera, Tail Ratio |
| **Portfolio** | MCR, PCR, Correlation | Tail Dependence, Absorption Ratio |

---

## Final Reminder

> **NOT FINANCIAL ADVICE**: All examples, metrics, and analyses in this documentation are for educational and research purposes only. They do not constitute investment advice or recommendations.
>
> Before making any investment decisions:
> 1. Consult a qualified financial professional
> 2. Conduct your own independent research
> 3. Understand that past performance does not guarantee future results
> 4. Consider your personal financial situation and risk tolerance
>
> The authors and contributors of NanuQuant assume no liability for any financial decisions made using this software.

---

## See Also

- [API Reference](api/core.md)
- [Mathematical Formulas](mathematics.md)
- [Testing Methodology](testing.md)
- [Full Disclaimer](https://github.com/launchstack-dev/nanuquant/blob/main/DISCLAIMER.md)
