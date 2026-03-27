# Advanced Trading Metrics API Reference

> **Disclaimer**: NanuQuant is for educational and research purposes only. These metrics do **NOT** constitute financial advice and should not be the sole basis for trading decisions. See [DISCLAIMER](https://github.com/launchstack-dev/nanuquant/blob/main/DISCLAIMER.md).

This document covers advanced trading metrics designed for systematic strategy evaluation.

Module: `nanuquant.advanced.trades`

---

## Table of Contents

- [Exposure Metrics](#exposure-metrics)
- [Return Adjustments](#return-adjustments)
- [Quality Metrics](#quality-metrics)
- [Risk Metrics](#risk-metrics)
- [Autocorrelation-Adjusted Metrics](#autocorrelation-adjusted-metrics)

---

## Exposure Metrics

### exposure

Percentage of time capital is deployed in the market.

```python
nq.exposure(returns: pl.Series) -> float
```

**Parameters:**
- `returns`: Series of period returns (0 indicates no position)

**Returns:** Fraction of periods with non-zero returns (0.0 to 1.0)

**Example:**
```python
returns = pl.Series([0.01, 0, 0.02, 0, 0, 0.01])
exp = nq.exposure(returns)  # 0.5 (50% of time in market)
```

**Use Case:** Useful for comparing strategies with different investment intensities. A strategy with high returns but low exposure may not utilize capital efficiently.

---

## Return Adjustments

### ghpr

Geometric Holding Period Return - the geometric mean of (1 + returns).

```python
nq.ghpr(returns: pl.Series) -> float
```

**Parameters:**
- `returns`: Series of period returns

**Returns:** Geometric mean return per period

**Note:** GHPR better represents compound growth than arithmetic mean, especially for volatile return series.

---

### rar

Risk-Adjusted Return - CAGR divided by market exposure.

```python
nq.rar(
    returns: pl.Series,
    periods_per_year: int = 252
) -> float
```

**Parameters:**
- `returns`: Series of period returns
- `periods_per_year`: Annualization factor

**Returns:** Annualized return per unit of exposure

**Use Case:** Compares strategies that are in the market for different amounts of time. A strategy in the market 50% of the time with 10% annual return has RAR of 20%.

---

## Quality Metrics

### cpc_index

Combines Profit Factor, Win Rate, and Payoff Ratio.

```python
nq.cpc_index(returns: pl.Series) -> float
```

**Formula:** `CPC = Profit Factor × Win Rate × Payoff Ratio`

**Returns:** CPC Index value

**Interpretation:**
- > 1.2: Strong strategy
- 0.8 - 1.2: Average strategy
- < 0.8: Weak strategy

---

### serenity_index

Measures the smoothness of equity growth relative to drawdowns.

```python
nq.serenity_index(returns: pl.Series) -> float
```

**Returns:** Serenity index (higher = smoother equity curve)

**Use Case:** Identifies strategies that produce steady returns with minimal drawdown periods.

---

### sqn

System Quality Number - measures the quality of a trading system.

```python
nq.sqn(returns: pl.Series) -> float
```

**Formula:** `SQN = √n × (mean return / std deviation)`

**Returns:** SQN value

**Interpretation (Van Tharp's guidelines):**
- 1.6 - 1.9: Below average
- 2.0 - 2.4: Average
- 2.5 - 2.9: Good
- 3.0 - 5.0: Excellent
- 5.0 - 6.9: Superb
- 7.0+: Holy Grail (be skeptical)

**Warning:** Very high SQN values may indicate overfitting or data issues.

---

### expectancy

Average expected return per trade.

```python
nq.expectancy(returns: pl.Series) -> float
```

**Formula:** `Expectancy = (Win Rate × Avg Win) + (Loss Rate × Avg Loss)`

**Returns:** Expected value per trade

**Use Case:** Determines if a strategy has positive expected value. Should be positive for a viable strategy.

---

### k_ratio

Measures the consistency of equity curve growth.

```python
nq.k_ratio(returns: pl.Series) -> float
```

**Returns:** K-Ratio value

**Interpretation:**
- Higher values indicate more consistent growth
- Penalizes equity curves with high variability around the trend

**Use Case:** Useful for identifying strategies with smooth, consistent performance vs. those with erratic returns.

---

## Risk Metrics

### risk_of_ruin

Probability of account depletion based on win rate and payoff ratio.

```python
nq.risk_of_ruin(
    returns: pl.Series,
    ruin_threshold: float = 0.5
) -> float
```

**Parameters:**
- `returns`: Series of period returns
- `ruin_threshold`: Fraction of capital loss considered "ruin" (default 50%)

**Returns:** Probability of reaching ruin threshold (0.0 to 1.0)

**Warning:** This is a simplified model that assumes:
- Independent returns
- Constant win rate and payoff ratio
- No position sizing changes

Real risk of ruin may be higher due to:
- Correlated losses
- Psychological factors
- Market regime changes

---

## Autocorrelation-Adjusted Metrics

Financial returns often exhibit autocorrelation (serial correlation), which can inflate traditional performance metrics. These adjusted metrics account for this effect.

### smart_sharpe

Sharpe Ratio adjusted for autocorrelation using Lo (2002) methodology.

```python
nq.smart_sharpe(
    returns: pl.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float
```

**Parameters:**
- `returns`: Series of period returns
- `risk_free_rate`: Annualized risk-free rate
- `periods_per_year`: Annualization factor

**Returns:** Autocorrelation-adjusted Sharpe ratio

**Note:** If returns are positively autocorrelated, traditional Sharpe overestimates risk-adjusted performance. Smart Sharpe provides a more conservative estimate.

**Reference:** Lo, A. W. (2002). "The Statistics of Sharpe Ratios." Financial Analysts Journal.

---

### smart_sortino

Sortino Ratio adjusted for autocorrelation.

```python
nq.smart_sortino(
    returns: pl.Series,
    risk_free_rate: float = 0.0,
    mar: float = 0.0,
    periods_per_year: int = 252
) -> float
```

**Parameters:**
- `returns`: Series of period returns
- `risk_free_rate`: Annualized risk-free rate
- `mar`: Minimum Acceptable Return
- `periods_per_year`: Annualization factor

**Returns:** Autocorrelation-adjusted Sortino ratio

---

### adjusted_sortino

Sortino Ratio adjusted for skewness and kurtosis.

```python
nq.adjusted_sortino(
    returns: pl.Series,
    risk_free_rate: float = 0.0,
    mar: float = 0.0,
    periods_per_year: int = 252
) -> float
```

**Returns:** Sortino ratio with higher moment adjustments

**Use Case:** Provides a more accurate risk-adjusted measure when returns are non-normal.

---

## Trade Conversion Module

Module: `nanuquant.trades`

For converting trade-level data to return series.

### trades_to_returns

Convert a DataFrame of trades to a return series.

```python
from nanuquant.trades import trades_to_returns, TradeConfig

config = TradeConfig(
    initial_capital=100000,
    return_method="percentage",
    aggregation_mode="daily"
)

returns = trades_to_returns(trades_df, config)
```

**Trade DataFrame Expected Columns:**
- `entry_date`: Trade entry date
- `exit_date`: Trade exit date
- `entry_price`: Entry price
- `exit_price`: Exit price
- `quantity`: Position size
- `direction`: "long" or "short"

---

## Usage Notes

### Combining Metrics

For comprehensive strategy evaluation, consider using multiple metrics:

```python
import nanuquant as nq

returns = pl.Series([...])  # Your strategy returns

# Quality assessment
sqn = nq.sqn(returns)
expectancy = nq.expectancy(returns)
k_ratio = nq.k_ratio(returns)

# Risk assessment
risk_ruin = nq.risk_of_ruin(returns)
smart_sr = nq.smart_sharpe(returns)

# Exposure efficiency
rar = nq.rar(returns)
exposure = nq.exposure(returns)

print(f"SQN: {sqn:.2f}")
print(f"Expectancy: {expectancy:.4f}")
print(f"K-Ratio: {k_ratio:.2f}")
print(f"Risk of Ruin: {risk_ruin:.2%}")
print(f"Smart Sharpe: {smart_sr:.2f}")
print(f"RAR: {rar:.2%}")
print(f"Exposure: {exposure:.2%}")
```

### Limitations

1. **Backtested results**: These metrics calculated on historical data do not guarantee future performance
2. **Overfitting**: High metric values may indicate overfitting to historical data
3. **Regime changes**: Metrics assume stable market conditions
4. **Transaction costs**: Unless explicitly included in returns, costs are not reflected
5. **Slippage**: Real execution may differ from backtested prices

---

## See Also

- [Core Metrics](core.md)
- [Institutional Metrics](institutional.md)
- [Mathematical Formulas](../mathematics.md)
- [Use Cases](../use-cases.md)
