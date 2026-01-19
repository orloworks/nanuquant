"""Robustness metrics for institutional portfolio evaluation.

This module provides statistically rigorous metrics for evaluating the robustness
of portfolio performance, including adjustments for multiple testing and
non-normality of returns.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import polars as pl
from scipy import stats as scipy_stats

from nanuquant.config import get_config
from nanuquant.core.distribution import kurtosis, skewness
from nanuquant.core.utils import get_annualization_factor, to_float_series
from nanuquant.core.validation import validate_min_length, validate_returns
from nanuquant.institutional._helpers import _expected_max_sharpe, newey_west_se


class PSRResult(NamedTuple):
    """Result of Probabilistic Sharpe Ratio calculation.

    Attributes
    ----------
    psr : float
        Probabilistic Sharpe Ratio (probability that true SR > benchmark).
    p_value : float
        p-value for the hypothesis test H0: SR <= benchmark_sr.
    """

    psr: float
    p_value: float


def probabilistic_sharpe_ratio(
    returns: pl.Series,
    benchmark_sr: float = 0.0,
    *,
    periods_per_year: int | None = None,
) -> PSRResult:
    """Calculate Probabilistic Sharpe Ratio (PSR).

    The PSR estimates the probability that the true Sharpe ratio exceeds
    a given benchmark, accounting for the estimation error and non-normality
    (skewness and kurtosis) of returns.

    Parameters
    ----------
    returns : pl.Series
        Period returns (not prices).
    benchmark_sr : float, default 0.0
        Benchmark Sharpe ratio to compare against. Default is 0 (no skill).
    periods_per_year : int, optional
        Number of periods per year for annualization. If None, uses config default.

    Returns
    -------
    PSRResult
        Named tuple with:
        - psr: Probability that true Sharpe > benchmark (0 to 1)
        - p_value: p-value for hypothesis test

    Notes
    -----
    The standard error of the Sharpe ratio is adjusted for non-normality:

    SE(SR) = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt-3)/4 * SR^2) / (n-1))

    The PSR is then: Phi((SR - SR*) / SE(SR))

    where Phi is the standard normal CDF and SR* is the benchmark.

    References
    ----------
    Bailey, D. H., & Lopez de Prado, M. (2012). The Sharpe ratio efficient
    frontier. Journal of Risk, 15(2), 3-44.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> returns = pl.Series(np.random.normal(0.001, 0.02, 252))
    >>> result = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
    >>> result.psr  # Probability true SR > 0
    0.85...
    >>> result.p_value
    0.14...
    """
    validate_returns(returns)
    validate_min_length(returns, 3, metric="probabilistic_sharpe_ratio")

    returns = to_float_series(returns)
    n = len(returns)

    config = get_config()
    ann_factor = get_annualization_factor(
        periods_per_year=periods_per_year or config.periods_per_year
    )

    # Calculate sample statistics
    mean_ret = returns.mean()
    std_ret = returns.std()

    if mean_ret is None or std_ret is None or std_ret == 0:
        if std_ret == 0 and mean_ret is not None:
            # For zero volatility, Sharpe Ratio is effectively infinite or -infinite.
            if mean_ret > 0:
                return PSRResult(psr=1.0, p_value=0.0)  # Infinite SR beats any finite benchmark
            if mean_ret < 0:
                return PSRResult(psr=0.0, p_value=1.0)  # -Infinite SR loses to any finite benchmark
            # mean_ret is 0, so SR is 0. Compare to benchmark.
            if benchmark_sr < 0:
                return PSRResult(psr=1.0, p_value=0.0)
            if benchmark_sr > 0:
                return PSRResult(psr=0.0, p_value=1.0)
        # Fallback for None values or if both SR and benchmark are 0
        return PSRResult(psr=0.5, p_value=0.5)

    # Sample Sharpe ratio (annualized)
    sr = float(mean_ret / std_ret) * math.sqrt(ann_factor)

    # Get skewness and excess kurtosis
    skew = skewness(returns)
    kurt = kurtosis(returns)  # This returns excess kurtosis

    # Standard error of Sharpe ratio adjusted for non-normality
    # Formula from Bailey & Lopez de Prado (2012)
    # SE = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt/4)*SR^2) / (n-1))
    # Note: kurt here is already excess kurtosis
    se_numerator = 1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2

    if se_numerator < 0:
        # Handle numerical edge cases
        se_numerator = abs(se_numerator)

    se_sr = math.sqrt(se_numerator / (n - 1))

    if se_sr == 0:
        # Perfect certainty (degenerate case)
        psr = 1.0 if sr > benchmark_sr else 0.0
        p_value = 0.0 if sr > benchmark_sr else 1.0
        return PSRResult(psr=psr, p_value=p_value)

    # Calculate test statistic
    z_score = (sr - benchmark_sr) / se_sr

    # PSR is the CDF of the test statistic
    psr = float(scipy_stats.norm.cdf(z_score))

    # p-value for one-sided test H0: SR <= benchmark_sr
    p_value = 1.0 - psr

    return PSRResult(psr=psr, p_value=p_value)


def deflated_sharpe_ratio(
    returns: pl.Series,
    n_trials: int,
    *,
    var_sharpe: float = 1.0,
    periods_per_year: int | None = None,
) -> float:
    """Calculate Deflated Sharpe Ratio (DSR).

    The DSR adjusts the Sharpe ratio for multiple testing by comparing
    against the expected maximum Sharpe from n independent trials under
    the null hypothesis of no skill.

    Parameters
    ----------
    returns : pl.Series
        Period returns (not prices).
    n_trials : int
        Number of independent trials/strategies tested.
        This represents the total number of backtests run during strategy
        development, including variations that were discarded.
    var_sharpe : float, default 1.0
        Variance of Sharpe ratio estimates under null hypothesis.
        Default assumes unit variance (standard case).
    periods_per_year : int, optional
        Number of periods per year for annualization. If None, uses config default.

    Returns
    -------
    float
        Deflated Sharpe Ratio: probability that true SR > expected max from
        n null trials. Values near 1.0 indicate genuine skill, while values
        near 0.0 suggest the observed Sharpe is likely due to overfitting.

    Notes
    -----
    The DSR uses the PSR framework but replaces the benchmark with the
    expected maximum Sharpe ratio from n independent trials under the null:

    E[max(SR_1, ..., SR_n)] ≈ sqrt(Var) * (sqrt(2*ln(n)) - (gamma + ln(ln(n) + ln(4*pi))) / (2*sqrt(2*ln(n))))

    where gamma is the Euler-Mascheroni constant.

    The DSR answers: "What is the probability that this strategy's Sharpe
    ratio is greater than what we'd expect from the best of n random strategies?"

    References
    ----------
    Bailey, D. H., & Lopez de Prado, M. (2014). The deflated Sharpe ratio:
    correcting for selection bias, backtest overfitting, and non-normality.
    Journal of Portfolio Management, 40(5), 94-107.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> # Strong strategy
    >>> returns = pl.Series(np.random.normal(0.002, 0.02, 252))
    >>> deflated_sharpe_ratio(returns, n_trials=100)
    0.75...

    >>> # After trying 1000 strategies, a lower DSR is expected
    >>> deflated_sharpe_ratio(returns, n_trials=1000)
    0.45...
    """
    validate_returns(returns)
    validate_min_length(returns, 3, metric="deflated_sharpe_ratio")

    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    if var_sharpe <= 0:
        raise ValueError(f"var_sharpe must be positive, got {var_sharpe}")

    # Calculate expected maximum Sharpe from n null trials
    expected_max_sr = _expected_max_sharpe(n_trials, var_sharpe)

    # Use PSR with expected max as the benchmark
    result = probabilistic_sharpe_ratio(
        returns,
        benchmark_sr=expected_max_sr,
        periods_per_year=periods_per_year,
    )

    return result.psr


def minimum_track_record_length(
    returns: pl.Series,
    target_sr: float = 0.0,
    confidence: float = 0.95,
    *,
    periods_per_year: int | None = None,
) -> int:
    """Calculate minimum track record length for statistical significance.

    Determines the minimum number of observations needed to conclude that
    the Sharpe ratio is statistically greater than a target with specified
    confidence.

    Parameters
    ----------
    returns : pl.Series
        Period returns (not prices). Used to estimate skewness and kurtosis.
    target_sr : float, default 0.0
        Target Sharpe ratio to exceed. Default is 0 (testing for positive SR).
    confidence : float, default 0.95
        Confidence level (0 to 1).
    periods_per_year : int, optional
        Number of periods per year for annualization. If None, uses config default.

    Returns
    -------
    int
        Minimum number of periods required.

    Notes
    -----
    Formula: n* = 1 + (1 + 0.5*SR^2 - skew*SR + (kurt/4)*SR^2) * (z_alpha / (SR - SR*))^2

    where z_alpha is the critical value for the confidence level.

    This is the Minimum Track Record Length (MinTRL) from Bailey & Lopez de Prado.

    References
    ----------
    Bailey, D. H., & Lopez de Prado, M. (2012). The Sharpe ratio efficient
    frontier. Journal of Risk, 15(2), 3-44.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> returns = pl.Series(np.random.normal(0.001, 0.02, 252))
    >>> minimum_track_record_length(returns, target_sr=0.0, confidence=0.95)
    156
    """
    validate_returns(returns)
    validate_min_length(returns, 3, metric="minimum_track_record_length")

    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    returns = to_float_series(returns)

    config = get_config()
    ann_factor = get_annualization_factor(
        periods_per_year=periods_per_year or config.periods_per_year
    )

    # Calculate sample statistics
    mean_ret = returns.mean()
    std_ret = returns.std()

    if mean_ret is None or std_ret is None or std_ret == 0:
        return len(returns)

    # Sample Sharpe ratio (annualized)
    sr = float(mean_ret / std_ret) * math.sqrt(ann_factor)

    # If sample SR <= target, we can never achieve significance
    if sr <= target_sr:
        return float("inf")  # type: ignore

    # Get skewness and excess kurtosis
    skew = skewness(returns)
    kurt = kurtosis(returns)

    # Critical value for confidence level
    z_alpha = float(scipy_stats.norm.ppf(confidence))

    # Non-normality adjustment factor
    adjustment = 1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2

    if adjustment < 0:
        adjustment = abs(adjustment)

    # Minimum track record length
    sr_diff = sr - target_sr
    min_trl = 1 + adjustment * (z_alpha / sr_diff) ** 2

    return int(math.ceil(min_trl))
