"""Core metrics modules.

This module provides the main metrics functions for quantitative finance.
"""

from nanuquant.core.distribution import (
    expected_return,
    geometric_mean,
    jarque_bera,
    kurtosis,
    outlier_loss_ratio,
    outlier_win_ratio,
    outliers,
    outliers_iqr,
    remove_outliers,
    remove_outliers_iqr,
    shapiro_wilk,
    skewness,
)
from nanuquant.core.rolling import (
    rolling_beta,
    rolling_greeks,
    rolling_sharpe,
    rolling_sortino,
    rolling_volatility,
)
from nanuquant.core.periods import (
    compare,
    distribution,
    monthly_returns,
)
from nanuquant.core.timeseries import (
    cumulative_returns,
    drawdown_details,
    equity_curve,
    histogram,
    yearly_returns,
)
from nanuquant.core.performance import (
    benchmark_correlation,
    calmar,
    common_sense_ratio,
    gain_to_pain_ratio,
    greeks,
    information_ratio,
    kelly_criterion,
    omega,
    r_squared,
    recovery_factor,
    risk_return_ratio,
    sharpe,
    sortino,
    tail_ratio,
    treynor_ratio,
    ulcer_performance_index,
)
from nanuquant.core.returns import (
    avg_loss,
    avg_return,
    avg_win,
    best,
    cagr,
    comp,
    consecutive_losses,
    consecutive_wins,
    payoff_ratio,
    profit_factor,
    win_rate,
    worst,
)
from nanuquant.core.risk import (
    cvar,
    downside_deviation,
    max_drawdown,
    to_drawdown_series,
    ulcer_index,
    var,
    volatility,
)
from nanuquant.core.utils import (
    compound_returns,
    get_annualization_factor,
    infer_frequency,
    log_returns,
    safe_divide,
    simple_returns,
    to_float_series,
)
from nanuquant.core.validation import (
    validate_benchmark_match,
    validate_min_length,
    validate_positive,
    validate_probability,
    validate_returns,
)

__all__ = [
    # Returns
    "comp",
    "cagr",
    "avg_return",
    "avg_win",
    "avg_loss",
    "best",
    "worst",
    "win_rate",
    "payoff_ratio",
    "profit_factor",
    "consecutive_wins",
    "consecutive_losses",
    # Risk
    "volatility",
    "var",
    "cvar",
    "max_drawdown",
    "to_drawdown_series",
    "ulcer_index",
    "downside_deviation",
    # Performance
    "sharpe",
    "sortino",
    "calmar",
    "omega",
    "gain_to_pain_ratio",
    "ulcer_performance_index",
    "kelly_criterion",
    "tail_ratio",
    "common_sense_ratio",
    "risk_return_ratio",
    "recovery_factor",
    # Benchmark metrics
    "greeks",
    "information_ratio",
    "r_squared",
    "treynor_ratio",
    "benchmark_correlation",
    # Distribution metrics
    "skewness",
    "kurtosis",
    "jarque_bera",
    "shapiro_wilk",
    "outlier_win_ratio",
    "outlier_loss_ratio",
    "expected_return",
    "geometric_mean",
    # Outlier detection
    "outliers",
    "remove_outliers",
    "outliers_iqr",
    "remove_outliers_iqr",
    # Rolling metrics
    "rolling_volatility",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_beta",
    "rolling_greeks",
    # Period analysis
    "monthly_returns",
    "distribution",
    "compare",
    # Timeseries (array/DataFrame outputs)
    "yearly_returns",
    "drawdown_details",
    "histogram",
    "cumulative_returns",
    "equity_curve",
    # Utils
    "compound_returns",
    "get_annualization_factor",
    "infer_frequency",
    "log_returns",
    "simple_returns",
    "safe_divide",
    "to_float_series",
    # Validation
    "validate_returns",
    "validate_min_length",
    "validate_benchmark_match",
    "validate_positive",
    "validate_probability",
]
