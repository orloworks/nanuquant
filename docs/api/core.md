# Core API Reference

> **Disclaimer**: NanuQuant is for educational and research purposes only. The metrics described here do **NOT** constitute financial advice. See [DISCLAIMER](https://github.com/launchstack-dev/nanuquant/blob/main/DISCLAIMER.md).

This document covers the core metrics modules in NanuQuant.

## Table of Contents

- [Returns Metrics](#returns-metrics)
- [Risk Metrics](#risk-metrics)
- [Performance Metrics](#performance-metrics)
- [Distribution Metrics](#distribution-metrics)
- [Rolling Metrics](#rolling-metrics)
- [Period Analysis](#period-analysis)
- [Timeseries Functions](#timeseries-functions)
- [Utility Functions](#utility-functions)

---

## Returns Metrics

Module: `nanuquant.core.returns`

### comp

Calculate the total compounded return.

```python
nq.comp(returns: pl.Series) -> float
```

**Parameters:**
- `returns`: Series of period returns

**Returns:** Total compounded return as a decimal (e.g., 0.25 = 25%)

**Example:**
```python
returns = pl.Series([0.10, 0.05, -0.02])
total = nq.comp(returns)  # 0.1319 (13.19%)
```

---

### cagr

Compound Annual Growth Rate - the annualized geometric mean return.

```python
nq.cagr(returns: pl.Series, periods_per_year: int = 252) -> float
```

**Parameters:**
- `returns`: Series of period returns
- `periods_per_year`: Number of periods per year (252 for daily, 12 for monthly)

**Returns:** Annualized return as a decimal

**Example:**
```python
returns = pl.Series([0.0004] * 252)  # 252 daily returns of 0.04%
annual_return = nq.cagr(returns)  # ~10.6% annualized
```

---

### avg_return

Arithmetic mean of returns. Zero-return periods are treated as periods without a
position and excluded by default, as in QuantStats.

```python
nq.avg_return(returns: pl.Series, *, exclude_zeros: bool = True) -> float
```

**Parameters:**
- `returns`: Series of period returns
- `exclude_zeros`: If False, average every period

**Returns:** Mean return

---

### avg_win

Average of positive returns only.

```python
nq.avg_win(returns: pl.Series) -> float
```

**Parameters:**
- `returns`: Series of period returns

**Returns:** Mean of returns > 0

---

### avg_loss

Average of negative returns only.

```python
nq.avg_loss(returns: pl.Series) -> float
```

**Parameters:**
- `returns`: Series of period returns

**Returns:** Mean of returns < 0 (as a negative number)

---

### best

Maximum single-period return.

```python
nq.best(returns: pl.Series) -> float
```

**Parameters:**
- `returns`: Series of period returns

**Returns:** Highest individual return

---

### worst

Minimum single-period return.

```python
nq.worst(returns: pl.Series) -> float
```

**Parameters:**
- `returns`: Series of period returns

**Returns:** Lowest individual return

---

### win_rate

Percentage of positive returns among non-zero periods, as in QuantStats.

```python
nq.win_rate(returns: pl.Series, *, exclude_zeros: bool = True) -> float
```

**Parameters:**
- `returns`: Series of period returns
- `exclude_zeros`: If False, divide by all periods instead of the non-zero ones

**Returns:** Proportion of counted periods > 0 (e.g., 0.55 = 55%)

---

### payoff_ratio

Ratio of average win to average loss magnitude.

```python
nq.payoff_ratio(returns: pl.Series) -> float
```

**Parameters:**
- `returns`: Series of period returns

**Returns:** |avg_win| / |avg_loss|

---

### profit_factor

Ratio of gross profits to gross losses.

```python
nq.profit_factor(returns: pl.Series) -> float
```

**Parameters:**
- `returns`: Series of period returns

**Returns:** Sum of positive returns / |Sum of negative returns|

---

### consecutive_wins

Maximum consecutive positive return periods.

```python
nq.consecutive_wins(returns: pl.Series) -> int
```

---

### consecutive_losses

Maximum consecutive negative return periods.

```python
nq.consecutive_losses(returns: pl.Series) -> int
```

---

## Risk Metrics

Module: `nanuquant.core.risk`

### volatility

Annualized standard deviation of returns.

```python
nq.volatility(
    returns: pl.Series,
    periods_per_year: int = 252,
    ddof: int = 1
) -> float
```

**Parameters:**
- `returns`: Series of period returns
- `periods_per_year`: Annualization factor
- `ddof`: Degrees of freedom (1 for sample std dev)

**Returns:** Annualized volatility as a decimal (e.g., 0.20 = 20%)

---

### var

Parametric Value at Risk assuming normal distribution.

```python
nq.var(
    returns: pl.Series,
    confidence: float = 0.95
) -> float
```

**Parameters:**
- `returns`: Series of period returns
- `confidence`: Confidence level (0.90, 0.95, or 0.99)

**Returns:** VaR as a negative decimal representing the return at the given confidence level (e.g., -0.02 means 2% loss)

**Note:** Parametric VaR assumes normally distributed returns. Real market returns often exhibit fat tails, making this a potentially optimistic estimate.

---

### cvar

Conditional Value at Risk (Expected Shortfall).

```python
nq.cvar(
    returns: pl.Series,
    confidence: float = 0.95
) -> float
```

**Parameters:**
- `returns`: Series of period returns
- `confidence`: Confidence level

**Returns:** Expected loss given that loss exceeds VaR

**Note:** CVaR is a coherent risk measure and generally preferred over VaR for risk management.

---

### max_drawdown

Maximum peak-to-trough decline.

```python
nq.max_drawdown(returns: pl.Series) -> float
```

**Parameters:**
- `returns`: Series of period returns

**Returns:** Maximum drawdown as a negative decimal (e.g., -0.25 = -25%)

Drawdown is measured from the starting capital: the running peak starts at 1.0, so a
series that has lost since its first period is in drawdown from the start. QuantStats
starts the peak at the first period's wealth; the two agree once the first return is
non-negative.

---

### to_drawdown_series

Convert returns to a drawdown series.

```python
nq.to_drawdown_series(returns: pl.Series) -> pl.Series
```

**Parameters:**
- `returns`: Series of period returns

**Returns:** Series named `drawdown`, the same length as `returns`: `wealth / peak - 1`,
where the peak is the running maximum of wealth including the starting capital (1.0).
Values are <= 0, null where the return is missing, and -1.0 from the first period at
which wealth reaches zero or below (ruin).

---

### ulcer_index

Measures depth and duration of drawdowns (Ulcer Index).

```python
nq.ulcer_index(returns: pl.Series) -> float
```

**Parameters:**
- `returns`: Series of period returns

**Returns:** Ulcer Index value (higher = worse drawdown profile)

---

### downside_deviation

Standard deviation of returns below the Minimum Acceptable Return (MAR).

```python
nq.downside_deviation(
    returns: pl.Series,
    mar: float = 0.0,
    periods_per_year: int = 252
) -> float
```

**Parameters:**
- `returns`: Series of period returns
- `mar`: Minimum Acceptable Return (default 0)
- `periods_per_year`: Annualization factor

**Returns:** Annualized downside deviation

---

## Performance Metrics

Module: `nanuquant.core.performance`

### sharpe

Sharpe Ratio - excess return per unit of total risk.

```python
nq.sharpe(
    returns: pl.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float
```

**Parameters:**
- `returns`: Series of period returns
- `risk_free_rate`: Annualized risk-free rate
- `periods_per_year`: Annualization factor

**Returns:** Sharpe ratio

**Note:** The Sharpe ratio assumes normally distributed returns. For non-normal returns, consider using adjusted measures.

---

### sortino

Sortino Ratio - excess return per unit of downside risk.

```python
nq.sortino(
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

**Returns:** Sortino ratio

---

### calmar

Calmar Ratio - CAGR divided by maximum drawdown.

```python
nq.calmar(
    returns: pl.Series,
    periods_per_year: int = 252
) -> float
```

**Parameters:**
- `returns`: Series of period returns
- `periods_per_year`: Annualization factor

**Returns:** Calmar ratio (positive value)

---

### omega

Omega Ratio - probability-weighted ratio of gains vs losses.

```python
nq.omega(
    returns: pl.Series,
    threshold: float = 0.0
) -> float
```

**Parameters:**
- `returns`: Series of period returns
- `threshold`: Return threshold (default 0)

**Returns:** Omega ratio (>1 indicates gains outweigh losses)

---

### gain_to_pain_ratio

Total gains divided by total loss magnitude.

```python
nq.gain_to_pain_ratio(returns: pl.Series) -> float
```

**Parameters:**
- `returns`: Series of period returns

**Returns:** Gain to pain ratio

---

### ulcer_performance_index

Return divided by Ulcer Index.

```python
nq.ulcer_performance_index(
    returns: pl.Series,
    periods_per_year: int = 252
) -> float
```

---

### kelly_criterion

Optimal position sizing based on win rate and payoff ratio.

```python
nq.kelly_criterion(returns: pl.Series) -> float
```

**Parameters:**
- `returns`: Series of period returns

**Formula:** `(R * W - (1 - W)) / R`, with `W` the win rate and `R` the payoff ratio.

**Returns:** Kelly fraction (optimal bet size as fraction of capital). With no losing
periods it equals the win rate; with no winning periods it is `-inf`.

**Note:** Full Kelly sizing is often too aggressive in practice. Many practitioners use fractional Kelly (e.g., half-Kelly).

---

### tail_ratio

Ratio of right tail magnitude to left tail magnitude.

```python
nq.tail_ratio(
    returns: pl.Series,
    percentile: float = 0.95
) -> float
```

**Parameters:**
- `returns`: Series of period returns
- `percentile`: Percentile for tail measurement

**Returns:** Absolute 95th percentile / Absolute 5th percentile

---

### common_sense_ratio

Combines return, win rate, and downside risk.

```python
nq.common_sense_ratio(returns: pl.Series) -> float
```

---

### risk_return_ratio

Inverse of Sharpe (volatility / return).

```python
nq.risk_return_ratio(returns: pl.Series) -> float
```

---

### recovery_factor

Total profit divided by maximum drawdown.

```python
nq.recovery_factor(returns: pl.Series) -> float
```

---

### greeks

Calculate Alpha and Beta relative to benchmark.

```python
nq.greeks(
    returns: pl.Series,
    benchmark: pl.Series,
    periods_per_year: int = 252
) -> tuple[float, float]
```

**Parameters:**
- `returns`: Strategy returns
- `benchmark`: Benchmark returns
- `periods_per_year`: Annualization factor

**Returns:** Tuple of (alpha, beta)

**Note:** Alpha and beta are estimated via linear regression. Past alpha does not guarantee future outperformance.

---

### information_ratio

Excess return over benchmark divided by tracking error.

```python
nq.information_ratio(
    returns: pl.Series,
    benchmark: pl.Series,
    periods_per_year: int = 252
) -> float
```

---

### r_squared

Coefficient of determination with benchmark.

```python
nq.r_squared(
    returns: pl.Series,
    benchmark: pl.Series
) -> float
```

---

### treynor_ratio

CAGR divided by Beta.

```python
nq.treynor_ratio(
    returns: pl.Series,
    benchmark: pl.Series,
    periods_per_year: int = 252
) -> float
```

---

### benchmark_correlation

Pearson correlation with benchmark.

```python
nq.benchmark_correlation(
    returns: pl.Series,
    benchmark: pl.Series
) -> float
```

---

## Distribution Metrics

Module: `nanuquant.core.distribution`

### skewness

Measure of distribution asymmetry.

```python
nq.skewness(returns: pl.Series) -> float
```

**Returns:**
- Positive: Right tail longer (large gains more likely)
- Negative: Left tail longer (large losses more likely)
- Zero: Symmetric distribution

---

### kurtosis

Excess kurtosis - measure of tail heaviness.

```python
nq.kurtosis(returns: pl.Series) -> float
```

**Returns:**
- Positive: Fat tails (extreme events more likely than normal)
- Negative: Thin tails
- Zero: Normal distribution

**Note:** Financial returns typically exhibit positive excess kurtosis (fat tails).

---

### jarque_bera

Jarque-Bera test for normality.

```python
nq.jarque_bera(returns: pl.Series) -> tuple[float, float]
```

**Returns:** Tuple of (test statistic, p-value)

**Interpretation:** Low p-value (< 0.05) rejects normality hypothesis.

---

### shapiro_wilk

Shapiro-Wilk test for normality.

```python
nq.shapiro_wilk(returns: pl.Series) -> tuple[float, float]
```

**Returns:** Tuple of (test statistic, p-value)

---

### outlier_win_ratio

The `quantile` return of the whole series divided by the mean non-negative return
(QuantStats definition).

```python
nq.outlier_win_ratio(returns: pl.Series, quantile: float = 0.99) -> float
```

---

### outlier_loss_ratio

The `quantile` return of the whole series divided by the mean negative return
(QuantStats definition).

```python
nq.outlier_loss_ratio(returns: pl.Series, quantile: float = 0.01) -> float
```

---

### expected_return

Geometric mean return per period, `prod(1 + r) ** (1 / n) - 1` (QuantStats definition).
For the arithmetic mean use `avg_return(..., exclude_zeros=False)`.

```python
nq.expected_return(returns: pl.Series) -> float
```

---

### geometric_mean

Geometric mean of (1 + returns).

```python
nq.geometric_mean(returns: pl.Series) -> float
```

---

### outliers

Identify outliers using IQR method.

```python
nq.outliers(returns: pl.Series, multiplier: float = 1.5) -> pl.Series
```

---

### remove_outliers

Remove outliers from series.

```python
nq.remove_outliers(returns: pl.Series, multiplier: float = 1.5) -> pl.Series
```

---

## Rolling Metrics

Module: `nanuquant.core.rolling`

### rolling_volatility

Rolling annualized volatility.

```python
nq.rolling_volatility(
    returns: pl.Series,
    *,
    rolling_period: int = 126,
    periods_per_year: int | None = None,
    annualize: bool = True
) -> pl.Series
```

All rolling metrics annualize with `periods_per_year` from the global config (252 by
default), like the core metrics; QuantStats' rolling functions default to 365, so pass
`periods_per_year=365` to reproduce them. Output keeps the input length; any window
containing a missing return is null.

---

### rolling_sharpe

Rolling Sharpe ratio.

```python
nq.rolling_sharpe(
    returns: pl.Series,
    *,
    risk_free_rate: float = 0.0,
    rolling_period: int = 126,
    periods_per_year: int | None = None,
    annualize: bool = True
) -> pl.Series
```

---

### rolling_sortino

Rolling Sortino ratio.

```python
nq.rolling_sortino(
    returns: pl.Series,
    *,
    risk_free_rate: float = 0.0,
    rolling_period: int = 126,
    periods_per_year: int | None = None,
    annualize: bool = True
) -> pl.Series
```

---

### rolling_beta

Rolling Beta to benchmark.

```python
nq.rolling_beta(
    returns: pl.Series,
    benchmark: pl.Series,
    *,
    rolling_period: int = 126
) -> pl.Series
```

---

### rolling_greeks

Rolling Alpha and Beta.

```python
nq.rolling_greeks(
    returns: pl.Series,
    benchmark: pl.Series,
    *,
    rolling_period: int = 126,
    periods_per_year: int | None = None
) -> pl.DataFrame  # columns: rolling_alpha, rolling_beta
```

**Returns:** DataFrame with `rolling_alpha` (annualized) and `rolling_beta` columns

---

## Period Analysis

Module: `nanuquant.core.periods`

### monthly_returns

Month-by-year return matrix.

```python
nq.monthly_returns(returns: pl.Series, dates: pl.Series) -> pl.DataFrame
```

---

### yearly_returns

Annual compounded returns.

```python
nq.yearly_returns(returns: pl.Series, dates: pl.Series) -> pl.DataFrame
```

---

### distribution

Return distribution by period type.

```python
nq.distribution(returns: pl.Series, period: str = "monthly") -> pl.DataFrame
```

---

## Timeseries Functions

Module: `nanuquant.core.timeseries`

### cumulative_returns

Cumulative wealth curve.

```python
from nanuquant.core.timeseries import cumulative_returns

curve = cumulative_returns(returns, starting_value=1.0)
```

---

### equity_curve

Equity growth over time with dates.

```python
from nanuquant.core.timeseries import equity_curve

eq = equity_curve(returns, dates, starting_value=10000)
```

---

### drawdown_details

Detailed information about top N drawdowns.

```python
from nanuquant.core.timeseries import drawdown_details

details = drawdown_details(returns, top_n=5)
```

**Returns:** DataFrame of the deepest `top_n` drawdown episodes, with start, valley and
end positions (or dates when `dates` is given), depth, length and recovery.

- **Start:** a drawdown that fell from the starting capital has a null `start`, the capital
  before the first period, and its `length` counts from there.
- **Positions and counts:** positions refer to the input series, while `length` and
  recovery count observed (non-missing) periods.
- **Unrecovered episodes:** an episode still open at the end has a null `recovery`.

---

## Utility Functions

Module: `nanuquant.core.utils`

### log_returns

Convert prices to log returns.

```python
from nanuquant.core.utils import log_returns

log_rets = log_returns(prices)
```

---

### simple_returns

Convert prices to simple percentage returns.

```python
from nanuquant.core.utils import simple_returns

pct_rets = simple_returns(prices)
```

---

### compound_returns

Convert simple returns to compound returns.

```python
from nanuquant.core.utils import compound_returns

comp_rets = compound_returns(simple_rets)
```

---

### infer_frequency

Auto-detect data frequency from dates.

```python
from nanuquant.core.utils import infer_frequency

freq = infer_frequency(dates)  # Returns "D", "W", "M", etc.
```

---

## See Also

- [Advanced Trading Metrics](advanced.md)
- [Institutional Metrics](institutional.md)
- [Mathematical Formulas](../mathematics.md)
