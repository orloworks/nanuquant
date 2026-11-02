# Indicators API Reference

> **Disclaimer**: NanuQuant is for educational and research purposes only. These indicators do **NOT** constitute financial advice and should not be the sole basis for trading decisions. See [DISCLAIMER](https://github.com/launchstack-dev/nanuquant/blob/main/DISCLAIMER.md).

Technical indicators on **price** series. Unlike the metrics modules, which take returns and
produce a scalar, every indicator here produces a series aligned with its input.

Module: `nanuquant.indicators`

---

## Inputs and outputs

Each function accepts either a `pl.Series` or a `pl.Expr` and returns the same kind:

```python
import polars as pl
from nanuquant import indicators as ind

prices = pl.Series("SPY", [100.0, 101.0, 102.5, 101.5, 103.0, 104.2])
ind.sma(prices, 3)                      # pl.Series named "SPY"

df = pl.DataFrame({"SPY": prices, "QQQ": prices * 2})
df.select(ind.rsi(pl.col("SPY", "QQQ"), 2))          # one pass over both columns
df.select(pl.col("SPY").ind.max_drawdown(3))         # `.ind` expression namespace
```

- A missing price (null **or** NaN) is missing, and so is a zero or negative price (bad data; as a divisor it would be an infinite return): the indicator is null there, and windowed indicators (`sma`, `max_drawdown`, rolling `current_drawdown`, and the return-window indicators) are null for any window that includes one. `cumulative_return` reads only the two endpoint prices, so a gap strictly inside its lookback does not null it. NaN never appears in the output.
- Output keeps the input column's name.
- Platform conventions are explicit keyword arguments, never silent defaults:
  `percent=` (x100 units), `annualize=` (volatility), `period=None` vs an integer
  (all-time vs rolling peak). A strategy engine that must match a platform, e.g. Composer,
  pins them per indicator.

---

## Functions

| Function | Formula | Notes |
|---|---|---|
| `current_price(p)` | `p_t` | NaN normalised to null |
| `sma(p, period)` | mean of the last `period` prices | |
| `ema(p, period, *, adjust=False)` | `y_t = (1-a) y_{t-1} + a p_t`, `a = 2/(period+1)`, seeded at the first price | Null until `period` prices seen. `adjust=False` is the full-history recursion (Composer convention) |
| `rsi(p, period, *, method="wilder", seed="zero")` | `100 - 100/(1 + G/L)`; `G`, `L` Wilder-smoothed gains/losses, `a = 1/period` | `seed="zero"`: smoothing starts at the first price, whose change counts as zero. `seed="first_change"`: starts at the first real change, as pandas-ta does; null at the first price. The two differ for roughly `3 * period` observations. No warm-up cut. After a gap the change is from the last valid price and the missing row is skipped (same as removing it). 100 when `L = 0`; null when `G = L = 0` or the price is missing |
| `cumulative_return(p, period, *, percent=False)` | `p_t / p_{t-period} - 1` | |
| `max_drawdown(p, period, *, percent=False)` | `max_u (P_u - p_u) / P_u` over the window, `P_u` the running peak *within* the window | Positive magnitude. Counts a mid-window spike and drop even after recovery |
| `current_drawdown(p, period=None, *, percent=False)` | `(P - p_t) / P`; `P` all-time running peak (`None`) or trailing `period` max | Positive magnitude |
| `std_dev_return(p, period, *, percent=False, annualize=False, periods_per_year=252)` | sample std (ddof=1) of the last `period` one-period returns | Exactly 0 on a flat window. Daily unless `annualize=True` |
| `moving_average_return(p, period, *, percent=False)` | mean of the last `period` one-period returns | |

For `std_dev_return` and `moving_average_return`, `period` counts **returns**, so a full
window needs `period + 1` prices.

One-period returns are `p_t / p_{t-1} - 1` with no forward-filling across a missing
price, so a gap makes the surrounding returns missing rather than spanning it.

---

## Composer-compatible settings

The settings a strategy engine uses to reproduce Composer's indicator values:

| Composer indicator | Call |
|---|---|
| Current price | `current_price(p)` |
| Moving average of price | `sma(p, n)` |
| Exponential moving average of price | `ema(p, n)` |
| Relative strength index | `rsi(p, n, seed="first_change")` |
| Cumulative return | `cumulative_return(p, n, percent=True)` |
| Max drawdown | `max_drawdown(p, n, percent=True)` |
| Standard deviation of return | `std_dev_return(p, n, percent=True)` |
| Moving average of return | `moving_average_return(p, n, percent=True)` |

These are pinned by differential tests against a reference implementation that has been
checked against Composer's own allocations (`tests/test_indicators.py`).

For RSI, `seed="first_change"` matches the Composer parser (pandas-ta). The seed only
matters while it hasn't decayed, roughly `3 * period` observations. On a short history and a
long period (RSI-200 on two years of prices) that is most of the series, and the two seeds
can pick different branches.
