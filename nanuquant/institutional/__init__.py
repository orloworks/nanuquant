"""Institutional-grade metrics for quantitative finance.

This module provides advanced metrics used by institutional investors that go
beyond standard QuantStats functionality, including robustness testing,
volatility modeling, systemic risk measures, and execution quality metrics.
"""

from nanuquant.institutional.execution import (
    ImplementationShortfallResult,
    execution_vwap,
    implementation_shortfall,
    market_impact_estimate,
    spread_cost,
    vwap_slippage,
)
from nanuquant.institutional.portfolio import (
    LedoitWolfResult,
    MCRResult,
    correlation_from_covariance,
    ledoit_wolf_covariance,
    marginal_contribution_to_risk,
    portfolio_volatility,
)
from nanuquant.institutional.robustness import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from nanuquant.institutional.systemic import (
    AbsorptionRatioResult,
    absorption_ratio,
    downside_correlation,
    lower_tail_dependence,
    upside_correlation,
)
from nanuquant.institutional.var_extensions import (
    cornish_fisher_var,
    entropic_var,
    historical_var,
    modified_var,
    parametric_var,
)
from nanuquant.institutional.volatility import (
    ARCHTestResult,
    GARCHResult,
    arch_effect_test,
    garch_volatility,
)

__all__ = [
    # Robustness metrics
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    # Volatility metrics
    "arch_effect_test",
    "garch_volatility",
    "ARCHTestResult",
    "GARCHResult",
    # Systemic metrics
    "absorption_ratio",
    "lower_tail_dependence",
    "downside_correlation",
    "upside_correlation",
    "AbsorptionRatioResult",
    # VaR extensions
    "cornish_fisher_var",
    "entropic_var",
    "historical_var",
    "modified_var",
    "parametric_var",
    # Portfolio metrics
    "marginal_contribution_to_risk",
    "ledoit_wolf_covariance",
    "portfolio_volatility",
    "correlation_from_covariance",
    "MCRResult",
    "LedoitWolfResult",
    # Execution metrics
    "implementation_shortfall",
    "market_impact_estimate",
    "vwap_slippage",
    "spread_cost",
    "execution_vwap",
    "ImplementationShortfallResult",
]
