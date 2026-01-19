"""Report generation modules.

This module provides functions for generating comprehensive metrics reports
and HTML tearsheets.
"""

from nanuquant.reports.html import (
    generate_html_report,
    save_html_report,
)
from nanuquant.reports.metrics import (
    MetricsReport,
    compute_benchmark_metrics,
    compute_distribution_metrics,
    compute_performance_metrics,
    compute_returns_metrics,
    compute_risk_metrics,
    compute_trading_metrics,
    full_metrics,
    metrics_summary,
)

__all__ = [
    # Main report functions
    "full_metrics",
    "metrics_summary",
    "MetricsReport",
    # Individual metric categories
    "compute_returns_metrics",
    "compute_risk_metrics",
    "compute_performance_metrics",
    "compute_distribution_metrics",
    "compute_trading_metrics",
    "compute_benchmark_metrics",
    # HTML reports
    "generate_html_report",
    "save_html_report",
]
