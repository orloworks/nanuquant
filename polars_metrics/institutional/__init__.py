"""Institutional metrics modules.

This module provides institutional-grade metrics for professional portfolio evaluation,
including Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).
"""

from polars_metrics.institutional.robustness import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)

__all__ = [
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
]
