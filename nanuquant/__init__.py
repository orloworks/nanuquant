"""
NanuQuant: Institutional quantitative analytics at Polars speed.

A complete QuantStats replacement with native Polars implementation and
institutional-grade robustness metrics.

This package provides:
- Core metrics: sharpe, sortino, volatility, max_drawdown, etc.
- Advanced trading metrics: exposure, ghpr, smart_sharpe, etc.
- Institutional metrics: deflated_sharpe_ratio, probabilistic_sharpe_ratio, etc.
- Rolling metrics: rolling_volatility, rolling_sharpe, rolling_sortino, etc.
- Report generation: HTML reports with metric summaries
- Trade conversion: Convert trade data to returns series

The package also registers a Polars expression namespace for idiomatic usage:

    >>> import polars as pl
    >>> import nanuquant as nq
    >>>
    >>> df = pl.DataFrame({"returns": [0.01, -0.02, 0.015, -0.01, 0.02] * 50})
    >>> df.select(pl.col("returns").metrics.sharpe())
    >>> df.select(pl.col("returns").metrics.max_drawdown())
"""

from nanuquant._version import __version__

# Import namespace to register it with Polars
from nanuquant.namespace import MetricsNamespace  # noqa: F401
from nanuquant.config import DEFAULT_CONFIG, MetricsConfig, get_config, set_config
from nanuquant.core import (
    # Returns
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
    # Risk
    cvar,
    downside_deviation,
    max_drawdown,
    to_drawdown_series,
    ulcer_index,
    var,
    volatility,
    # Performance
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
    # Distribution
    expected_return,
    geometric_mean,
    jarque_bera,
    kurtosis,
    outlier_loss_ratio,
    outlier_win_ratio,
    shapiro_wilk,
    skewness,
    # Rolling
    rolling_beta,
    rolling_greeks,
    rolling_sharpe,
    rolling_sortino,
    rolling_volatility,
    # Outlier detection
    outliers,
    outliers_iqr,
    remove_outliers,
    remove_outliers_iqr,
    # Period analysis
    compare,
    distribution,
    monthly_returns,
    # Timeseries (array/DataFrame outputs)
    cumulative_returns,
    drawdown_details,
    equity_curve,
    histogram,
    yearly_returns,
    # Utils (return conversion)
    compound_returns,
    log_returns,
    simple_returns,
)
from nanuquant.advanced import (
    # Trading metrics
    adjusted_sortino,
    cpc_index,
    expectancy,
    exposure,
    ghpr,
    k_ratio,
    rar,
    risk_of_ruin,
    serenity_index,
    smart_sharpe,
    smart_sortino,
    sqn,
)
from nanuquant.institutional import (
    # Robustness metrics
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from nanuquant.exceptions import (
    BenchmarkMismatchError,
    EmptySeriesError,
    InsufficientDataError,
    InvalidFrequencyError,
    MetricsError,
)
from nanuquant.reports import (
    MetricsReport,
    full_metrics,
    generate_html_report,
    metrics_summary,
    save_html_report,
)
from nanuquant.trades import (
    Trade,
    TradeConfig,
    TradeResult,
    calculate_single_trade_return,
    trades_to_returns,
)

__all__ = [
    "__version__",
    # Namespace plugin (registered automatically on import)
    "MetricsNamespace",
    # Config
    "MetricsConfig",
    "DEFAULT_CONFIG",
    "get_config",
    "set_config",
    # Exceptions
    "MetricsError",
    "EmptySeriesError",
    "InsufficientDataError",
    "BenchmarkMismatchError",
    "InvalidFrequencyError",
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
    # Rolling metrics
    "rolling_volatility",
    "rolling_sharpe",
    "rolling_sortino",
    "rolling_beta",
    "rolling_greeks",
    # Outlier detection
    "outliers",
    "remove_outliers",
    "outliers_iqr",
    "remove_outliers_iqr",
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
    # Utils (return conversion)
    "log_returns",
    "simple_returns",
    "compound_returns",
    # Trading metrics
    "exposure",
    "ghpr",
    "rar",
    "cpc_index",
    "serenity_index",
    "risk_of_ruin",
    "adjusted_sortino",
    "smart_sharpe",
    "smart_sortino",
    "sqn",
    "expectancy",
    "k_ratio",
    # Institutional / Robustness metrics
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    # Reports
    "MetricsReport",
    "full_metrics",
    "metrics_summary",
    "generate_html_report",
    "save_html_report",
    # Trade conversion
    "trades_to_returns",
    "calculate_single_trade_return",
    "Trade",
    "TradeResult",
    "TradeConfig",
]
