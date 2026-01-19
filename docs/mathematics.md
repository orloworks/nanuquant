# Mathematical Foundations

> **Disclaimer**: NanuQuant is for educational and research purposes only. Understanding these formulas does **NOT** constitute financial advice. Mathematical models have limitations - see [Statistical Assumptions](#statistical-assumptions-and-limitations). See [DISCLAIMER.md](../DISCLAIMER.md).

This document details the mathematical formulas implemented in NanuQuant.

---

## Table of Contents

- [Return Calculations](#return-calculations)
- [Risk Metrics](#risk-metrics)
- [Performance Ratios](#performance-ratios)
- [Distribution Statistics](#distribution-statistics)
- [Advanced Trading Metrics](#advanced-trading-metrics)
- [Institutional Metrics](#institutional-metrics)
- [Statistical Assumptions and Limitations](#statistical-assumptions-and-limitations)

---

## Return Calculations

### Simple Returns

Simple (percentage) returns from prices:

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$

Where:
- $r_t$ = Return at time t
- $P_t$ = Price at time t

### Log Returns

Continuously compounded returns:

$$r_t^{log} = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})$$

**Property:** Log returns are additive: $r_{t_1 \to t_n}^{log} = \sum_{i=1}^{n} r_{t_i}^{log}$

### Compounded Return

Total return from compounding period returns:

$$R_{total} = \prod_{t=1}^{n}(1 + r_t) - 1$$

### Compound Annual Growth Rate (CAGR)

Annualized geometric mean return:

$$CAGR = \left(\prod_{t=1}^{n}(1 + r_t)\right)^{\frac{k}{n}} - 1$$

Where:
- $n$ = Number of periods
- $k$ = Periods per year (252 for daily, 12 for monthly)

**Alternative form:**

$$CAGR = \left(1 + R_{total}\right)^{\frac{k}{n}} - 1$$

### Geometric Mean

Per-period geometric mean return:

$$G = \left(\prod_{t=1}^{n}(1 + r_t)\right)^{\frac{1}{n}} - 1$$

---

## Risk Metrics

### Volatility (Standard Deviation)

Annualized standard deviation of returns:

$$\sigma_{annual} = \sigma \cdot \sqrt{k}$$

Where:

$$\sigma = \sqrt{\frac{1}{n-1}\sum_{t=1}^{n}(r_t - \bar{r})^2}$$

- $\bar{r}$ = Mean return
- $k$ = Periods per year

### Downside Deviation

Standard deviation of returns below a threshold (MAR):

$$\sigma_d = \sqrt{\frac{1}{n}\sum_{t=1}^{n}\min(r_t - MAR, 0)^2}$$

Where MAR = Minimum Acceptable Return (often 0 or the risk-free rate)

### Value at Risk (VaR)

**Parametric VaR** (assuming normal distribution):

$$VaR_\alpha = -(\mu + z_\alpha \cdot \sigma)$$

Where:
- $\mu$ = Mean return
- $\sigma$ = Standard deviation
- $z_\alpha$ = Z-score for confidence level (e.g., -1.645 for 95%)

**Historical VaR:**

$$VaR_\alpha = -\text{Percentile}_{1-\alpha}(r_1, r_2, ..., r_n)$$

### Conditional VaR (Expected Shortfall)

Expected loss given that the loss exceeds VaR:

$$CVaR_\alpha = -E[r | r < -VaR_\alpha]$$

**For normal distribution:**

$$CVaR_\alpha = \mu + \sigma \cdot \frac{\phi(z_\alpha)}{\Phi(z_\alpha)}$$

Where $\phi$ and $\Phi$ are the PDF and CDF of the standard normal.

### Cornish-Fisher VaR

Adjusts parametric VaR for skewness and kurtosis:

$$z_{CF} = z + \frac{(z^2 - 1)S}{6} + \frac{(z^3 - 3z)K}{24} - \frac{(2z^3 - 5z)S^2}{36}$$

Where:
- $z$ = Standard normal quantile
- $S$ = Skewness
- $K$ = Excess kurtosis

Then: $VaR_{CF} = -(\mu + z_{CF} \cdot \sigma)$

### Maximum Drawdown

$$MDD = \min_{t \in [0,T]} \left(\frac{P_t}{\max_{s \in [0,t]} P_s} - 1\right)$$

Or equivalently, using a cumulative return series:

$$MDD = \min_{t}\left(\frac{C_t}{\max_{s \leq t} C_s} - 1\right)$$

### Ulcer Index

Root mean square of drawdowns:

$$UI = \sqrt{\frac{1}{n}\sum_{t=1}^{n}D_t^2}$$

Where $D_t$ = Drawdown at time t (as a positive percentage)

---

## Performance Ratios

### Sharpe Ratio

Excess return per unit of total risk:

$$SR = \frac{\bar{r}_{excess}}{\sigma} \cdot \sqrt{k} = \frac{CAGR - r_f}{\sigma_{annual}}$$

Where:
- $\bar{r}_{excess}$ = Mean excess return over risk-free rate
- $r_f$ = Risk-free rate (annualized)

### Sortino Ratio

Excess return per unit of downside risk:

$$Sortino = \frac{CAGR - r_f}{\sigma_d}$$

Where $\sigma_d$ = Annualized downside deviation

### Calmar Ratio

CAGR relative to maximum drawdown:

$$Calmar = \frac{CAGR}{|MDD|}$$

### Omega Ratio

Probability-weighted ratio of gains to losses:

$$\Omega(\theta) = \frac{\int_\theta^\infty (1 - F(r)) dr}{\int_{-\infty}^\theta F(r) dr}$$

Where:
- $F(r)$ = CDF of returns
- $\theta$ = Threshold return (usually 0)

**Discrete approximation:**

$$\Omega = \frac{\sum_{r_t > \theta}(r_t - \theta)}{\sum_{r_t < \theta}(\theta - r_t)}$$

### Information Ratio

Excess return over benchmark per unit of tracking error:

$$IR = \frac{R_p - R_b}{\sigma_{tracking}}$$

Where:
- $R_p$ = Portfolio return (annualized)
- $R_b$ = Benchmark return (annualized)
- $\sigma_{tracking} = \sigma(r_p - r_b) \cdot \sqrt{k}$ = Annualized tracking error

### Treynor Ratio

Excess return per unit of systematic risk:

$$Treynor = \frac{CAGR - r_f}{\beta}$$

### Alpha and Beta (CAPM)

From linear regression of returns against benchmark:

$$r_t = \alpha + \beta \cdot r_{benchmark,t} + \epsilon_t$$

- $\beta$ = Systematic risk (sensitivity to benchmark)
- $\alpha$ = Excess return after adjusting for beta (annualized)

$$\beta = \frac{Cov(r, r_b)}{Var(r_b)}$$

$$\alpha = (\bar{r} - r_f) - \beta \cdot (\bar{r}_b - r_f)$$

### Kelly Criterion

Optimal position sizing:

$$f^* = \frac{p \cdot b - q}{b} = \frac{p}{q} - \frac{1}{b}$$

Where:
- $p$ = Probability of win (win rate)
- $q$ = Probability of loss = 1 - p
- $b$ = Win/loss ratio (payoff ratio)

**Alternative (continuous) form:**

$$f^* = \frac{\mu - r_f}{\sigma^2}$$

---

## Distribution Statistics

### Skewness

Measure of asymmetry (third standardized moment):

$$Skew = \frac{E[(r - \mu)^3]}{\sigma^3} = \frac{\frac{1}{n}\sum(r_t - \bar{r})^3}{\left(\frac{1}{n}\sum(r_t - \bar{r})^2\right)^{3/2}}$$

- Positive: Right tail longer (large gains more likely)
- Negative: Left tail longer (large losses more likely)

### Excess Kurtosis

Measure of tail heaviness (fourth standardized moment - 3):

$$Kurt = \frac{E[(r - \mu)^4]}{\sigma^4} - 3$$

- Positive: Fat tails (extreme events more likely than normal)
- Zero: Normal distribution
- Negative: Thin tails

### Jarque-Bera Test

Test for normality based on skewness and kurtosis:

$$JB = \frac{n}{6}\left(S^2 + \frac{K^2}{4}\right)$$

Under null hypothesis of normality: $JB \sim \chi^2_2$

---

## Advanced Trading Metrics

### System Quality Number (SQN)

$$SQN = \sqrt{n} \cdot \frac{\bar{r}}{\sigma}$$

Where $n$ = Number of trades/periods

### Expectancy

$$E = p \cdot \bar{W} + (1-p) \cdot \bar{L}$$

Where:
- $p$ = Win rate
- $\bar{W}$ = Average win
- $\bar{L}$ = Average loss (negative)

### CPC Index

$$CPC = PF \times WR \times PR$$

Where:
- $PF$ = Profit Factor = $\frac{\sum \text{wins}}{|\sum \text{losses}|}$
- $WR$ = Win Rate
- $PR$ = Payoff Ratio = $\frac{|\bar{W}|}{|\bar{L}|}$

### Smart Sharpe (Lo Adjustment)

Adjusts for autocorrelation:

$$SR_{adj} = SR \times \sqrt{\frac{1}{1 + \sum_{k=1}^{q}(2 - \frac{2k}{q+1})\rho_k}}$$

Where $\rho_k$ = Autocorrelation at lag k

**Reference:** Lo, A. W. (2002). "The Statistics of Sharpe Ratios."

### K-Ratio

Measures consistency of equity curve:

$$K = \frac{\text{Slope of equity curve}}{\text{Standard error of slope}}$$

A regression of cumulative returns against time is performed, and K-ratio is the t-statistic of the slope.

### Risk of Ruin

Simple approximation:

$$RoR = \left(\frac{1 - Edge}{1 + Edge}\right)^{Units}$$

Where:
- $Edge = p \cdot \frac{W}{L} - (1-p)$
- $Units$ = Number of "betting units" until ruin

---

## Institutional Metrics

### Probabilistic Sharpe Ratio (PSR)

Probability that true Sharpe exceeds a benchmark:

$$PSR(\hat{SR}^*) = \Phi\left(\frac{(\hat{SR} - SR^*)\sqrt{n-1}}{\sqrt{1 - \hat{\gamma}_3 \hat{SR} + \frac{\hat{\gamma}_4 - 1}{4}\hat{SR}^2}}\right)$$

Where:
- $\hat{SR}$ = Observed Sharpe ratio
- $SR^*$ = Benchmark Sharpe ratio
- $\hat{\gamma}_3$ = Sample skewness
- $\hat{\gamma}_4$ = Sample kurtosis
- $n$ = Number of observations

**Reference:** Bailey, D. H., & López de Prado, M. (2012).

### Deflated Sharpe Ratio (DSR)

Adjusts PSR for multiple testing:

$$DSR = PSR(SR^*_0)$$

Where $SR^*_0$ is the expected maximum Sharpe under null:

$$E[\max\{SR_1, ..., SR_N\}] \approx \sqrt{V[\{SR_k\}]} \cdot \left((1-\gamma)Z^{-1}(1-\frac{1}{N}) + \gamma Z^{-1}(1-\frac{1}{N}e^{-1})\right)$$

### Minimum Track Record Length

$$MinTRL = 1 + \left[1 - \hat{\gamma}_3 \hat{SR} + \frac{\hat{\gamma}_4-1}{4}\hat{SR}^2\right] \times \left(\frac{Z_\alpha}{\hat{SR} - SR^*}\right)^2$$

### GARCH(1,1) Model

Conditional variance model:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

Where:
- $\omega$ = Constant (long-run variance weight)
- $\alpha$ = ARCH coefficient (shock impact)
- $\beta$ = GARCH coefficient (persistence)
- $\epsilon_t = r_t - \mu$ = Return shock

**Persistence:** $\alpha + \beta$ (should be < 1 for stationarity)

**Long-run variance:** $\sigma^2_{LR} = \frac{\omega}{1 - \alpha - \beta}$

### Ledoit-Wolf Shrinkage

Shrunk covariance estimator:

$$\Sigma_{shrunk} = \delta \cdot F + (1-\delta) \cdot S$$

Where:
- $S$ = Sample covariance matrix
- $F$ = Structured target (e.g., diagonal)
- $\delta$ = Optimal shrinkage intensity

### Absorption Ratio

Systemic risk measure:

$$AR = \frac{\sum_{i=1}^{k}\lambda_i}{\sum_{j=1}^{n}\lambda_j}$$

Where:
- $\lambda_i$ = Eigenvalues from PCA of returns
- $k$ = Number of top principal components (e.g., 5)
- $n$ = Total number of assets

### Implementation Shortfall

$$IS = \text{Paper P\&L} - \text{Realized P\&L}$$

Decomposition:
- **Delay cost**: Price movement from decision to execution start
- **Market impact**: Price movement due to execution
- **Timing cost**: Residual costs

### Square-Root Market Impact Model

$$Impact = k \cdot \sigma \cdot \sqrt{\frac{V}{ADV}}$$

Where:
- $k$ = Market impact coefficient (~0.1-0.5)
- $\sigma$ = Volatility
- $V$ = Trade volume
- $ADV$ = Average daily volume

---

## Statistical Assumptions and Limitations

### Common Assumptions

Most financial metrics assume one or more of:

1. **Independent returns**: Each return is independent of previous returns
2. **Identically distributed**: All returns come from the same distribution
3. **Normal distribution**: Returns follow a Gaussian distribution
4. **Stationarity**: Statistical properties don't change over time

### Reality Check

**Financial returns typically exhibit:**

| Assumption | Reality |
|------------|---------|
| Independence | Autocorrelation (momentum, mean reversion) |
| Identical distribution | Time-varying volatility (GARCH effects) |
| Normal distribution | Fat tails (excess kurtosis), negative skew |
| Stationarity | Regime changes, structural breaks |

### Implications

1. **Parametric VaR** underestimates tail risk
2. **Sharpe ratio** may be inflated by autocorrelation
3. **Historical statistics** may not predict future behavior
4. **Backtests** often overfit to historical patterns

### Best Practices

1. **Test for normality** using Jarque-Bera or Shapiro-Wilk
2. **Test for ARCH effects** if modeling volatility
3. **Use robust metrics** (CVaR, Cornish-Fisher VaR, Smart Sharpe)
4. **Validate with PSR/DSR** to assess statistical significance
5. **Never rely on a single metric** - use multiple perspectives

---

## References

1. Sharpe, W. F. (1966). "Mutual Fund Performance." Journal of Business.
2. Sortino, F. A., & van der Meer, R. (1991). "Downside Risk." Journal of Portfolio Management.
3. Lo, A. W. (2002). "The Statistics of Sharpe Ratios." Financial Analysts Journal.
4. Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." Journal of Risk.
5. Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio." Journal of Portfolio Management.
6. Ledoit, O., & Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices."
7. Kritzman, M., et al. (2011). "Principal Components as a Measure of Systemic Risk." Journal of Portfolio Management.
8. Engle, R. F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation." Econometrica.

---

## See Also

- [API Reference](api/core.md)
- [Testing Methodology](testing.md)
- [Use Cases](use-cases.md)
