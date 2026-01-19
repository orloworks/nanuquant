"""Comprehensive metrics report generation.

This module provides functions to compute all metrics at once,
similar to QuantStats' full_metrics functionality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from nanuquant.config import get_config
from nanuquant.core import (
    avg_loss,
    avg_return,
    avg_win,
    best,
    cagr,
    calmar,
    common_sense_ratio,
    comp,
    consecutive_losses,
    consecutive_wins,
    cvar,
    downside_deviation,
    expected_return,
    gain_to_pain_ratio,
    geometric_mean,
    greeks,
    information_ratio,
    jarque_bera,
    kelly_criterion,
    kurtosis,
    max_drawdown,
    omega,
    outlier_loss_ratio,
    outlier_win_ratio,
    payoff_ratio,
    profit_factor,
    r_squared,
    recovery_factor,
    risk_return_ratio,
    shapiro_wilk,
    sharpe,
    skewness,
    sortino,
    tail_ratio,
    treynor_ratio,
    ulcer_index,
    ulcer_performance_index,
    var,
    volatility,
    win_rate,
    worst,
)
from nanuquant.advanced import (
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


@dataclass
class MetricsReport:
    """Container for comprehensive metrics report.

    Attributes
    ----------
    returns_metrics : dict
        Basic return statistics.
    risk_metrics : dict
        Risk metrics (volatility, VaR, drawdown).
    performance_metrics : dict
        Performance ratios (Sharpe, Sortino, etc.).
    distribution_metrics : dict
        Distribution statistics (skew, kurtosis).
    trading_metrics : dict
        Advanced trading metrics.
    benchmark_metrics : dict, optional
        Benchmark comparison metrics (if benchmark provided).
    """

    returns_metrics: dict[str, Any]
    risk_metrics: dict[str, Any]
    performance_metrics: dict[str, Any]
    distribution_metrics: dict[str, Any]
    trading_metrics: dict[str, Any]
    benchmark_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        result = {
            "returns": self.returns_metrics,
            "risk": self.risk_metrics,
            "performance": self.performance_metrics,
            "distribution": self.distribution_metrics,
            "trading": self.trading_metrics,
        }
        if self.benchmark_metrics is not None:
            result["benchmark"] = self.benchmark_metrics
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_polars(self) -> pl.DataFrame:
        """Convert report to Polars DataFrame.

        Returns a flat DataFrame with category and metric columns.
        """
        rows = []
        for category, metrics in self.to_dict().items():
            for metric_name, value in metrics.items():
                rows.append(
                    {
                        "category": category,
                        "metric": metric_name,
                        "value": float(value) if value is not None else None,
                    }
                )
        return pl.DataFrame(rows)


def compute_returns_metrics(
    returns: pl.Series,
    *,
    periods_per_year: int | None = None,
) -> dict[str, Any]:
    """Compute basic return statistics.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    periods_per_year : int, optional
        Periods per year for annualization.

    Returns
    -------
    dict
        Dictionary of return metrics.
    """
    config = get_config()
    ppy = periods_per_year or config.periods_per_year

    return {
        "total_return": comp(returns),
        "cagr": cagr(returns, periods_per_year=ppy),
        "avg_return": avg_return(returns),
        "avg_win": avg_win(returns),
        "avg_loss": avg_loss(returns),
        "best": best(returns),
        "worst": worst(returns),
        "win_rate": win_rate(returns),
        "payoff_ratio": payoff_ratio(returns),
        "profit_factor": profit_factor(returns),
        "consecutive_wins": consecutive_wins(returns),
        "consecutive_losses": consecutive_losses(returns),
    }


def compute_risk_metrics(
    returns: pl.Series,
    *,
    periods_per_year: int | None = None,
    var_confidence: float | None = None,
) -> dict[str, Any]:
    """Compute risk metrics.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    periods_per_year : int, optional
        Periods per year for annualization.
    var_confidence : float, optional
        Confidence level for VaR/CVaR.

    Returns
    -------
    dict
        Dictionary of risk metrics.
    """
    config = get_config()
    ppy = periods_per_year or config.periods_per_year
    confidence = var_confidence or config.var_confidence

    return {
        "volatility": volatility(returns, periods_per_year=ppy),
        "var": var(returns, confidence=confidence),
        "cvar": cvar(returns, confidence=confidence),
        "max_drawdown": max_drawdown(returns),
        "ulcer_index": ulcer_index(returns),
        "downside_deviation": downside_deviation(returns, periods_per_year=ppy),
    }


def compute_performance_metrics(
    returns: pl.Series,
    *,
    risk_free_rate: float | None = None,
    periods_per_year: int | None = None,
) -> dict[str, Any]:
    """Compute performance ratios.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    risk_free_rate : float, optional
        Annualized risk-free rate.
    periods_per_year : int, optional
        Periods per year for annualization.

    Returns
    -------
    dict
        Dictionary of performance metrics.
    """
    config = get_config()
    rf = risk_free_rate if risk_free_rate is not None else config.risk_free_rate
    ppy = periods_per_year or config.periods_per_year

    return {
        "sharpe": sharpe(returns, risk_free_rate=rf, periods_per_year=ppy),
        "sortino": sortino(returns, risk_free_rate=rf, periods_per_year=ppy),
        "calmar": calmar(returns, periods_per_year=ppy),
        "omega": omega(returns, risk_free_rate=rf, periods_per_year=ppy),
        "gain_to_pain_ratio": gain_to_pain_ratio(returns),
        "ulcer_performance_index": ulcer_performance_index(
            returns, risk_free_rate=rf
        ),
        "kelly_criterion": kelly_criterion(returns),
        "tail_ratio": tail_ratio(returns),
        "common_sense_ratio": common_sense_ratio(returns),
        "risk_return_ratio": risk_return_ratio(returns),
        "recovery_factor": recovery_factor(returns),
    }


def compute_distribution_metrics(returns: pl.Series) -> dict[str, Any]:
    """Compute distribution statistics.

    Parameters
    ----------
    returns : pl.Series
        Period returns.

    Returns
    -------
    dict
        Dictionary of distribution metrics.
    """
    jb_stat, jb_pvalue = jarque_bera(returns)
    sw_stat, sw_pvalue = shapiro_wilk(returns)

    return {
        "skewness": skewness(returns),
        "kurtosis": kurtosis(returns),
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_pvalue,
        "shapiro_wilk_stat": sw_stat,
        "shapiro_wilk_pvalue": sw_pvalue,
        "outlier_win_ratio": outlier_win_ratio(returns),
        "outlier_loss_ratio": outlier_loss_ratio(returns),
        "expected_return": expected_return(returns),
        "geometric_mean": geometric_mean(returns),
    }


def compute_trading_metrics(
    returns: pl.Series,
    *,
    risk_free_rate: float | None = None,
    periods_per_year: int | None = None,
) -> dict[str, Any]:
    """Compute advanced trading metrics.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    risk_free_rate : float, optional
        Annualized risk-free rate.
    periods_per_year : int, optional
        Periods per year for annualization.

    Returns
    -------
    dict
        Dictionary of trading metrics.
    """
    config = get_config()
    rf = risk_free_rate if risk_free_rate is not None else config.risk_free_rate
    ppy = periods_per_year or config.periods_per_year

    return {
        "exposure": exposure(returns),
        "ghpr": ghpr(returns),
        "rar": rar(returns),
        "cpc_index": cpc_index(returns),
        "serenity_index": serenity_index(returns),
        "risk_of_ruin": risk_of_ruin(returns),
        "adjusted_sortino": adjusted_sortino(
            returns, risk_free_rate=rf, periods_per_year=ppy
        ),
        "smart_sharpe": smart_sharpe(returns, risk_free_rate=rf, periods_per_year=ppy),
        "smart_sortino": smart_sortino(
            returns, risk_free_rate=rf, periods_per_year=ppy
        ),
        "sqn": sqn(returns),
        "expectancy": expectancy(returns),
        "k_ratio": k_ratio(returns),
    }


def compute_benchmark_metrics(
    returns: pl.Series,
    benchmark: pl.Series,
    *,
    risk_free_rate: float | None = None,
    periods_per_year: int | None = None,
) -> dict[str, Any]:
    """Compute benchmark comparison metrics.

    Parameters
    ----------
    returns : pl.Series
        Strategy period returns.
    benchmark : pl.Series
        Benchmark period returns.
    risk_free_rate : float, optional
        Annualized risk-free rate.
    periods_per_year : int, optional
        Periods per year for annualization.

    Returns
    -------
    dict
        Dictionary of benchmark metrics.
    """
    config = get_config()
    rf = risk_free_rate if risk_free_rate is not None else config.risk_free_rate
    ppy = periods_per_year or config.periods_per_year

    alpha, beta = greeks(returns, benchmark, periods_per_year=ppy)

    return {
        "alpha": alpha,
        "beta": beta,
        "information_ratio": information_ratio(returns, benchmark),
        "r_squared": r_squared(returns, benchmark),
        "treynor_ratio": treynor_ratio(
            returns, benchmark, risk_free_rate=rf, periods_per_year=ppy
        ),
    }


def full_metrics(
    returns: pl.Series,
    benchmark: pl.Series | None = None,
    *,
    risk_free_rate: float | None = None,
    periods_per_year: int | None = None,
    var_confidence: float | None = None,
) -> MetricsReport:
    """Compute comprehensive metrics report.

    This function calculates all available metrics for a return series,
    similar to QuantStats' full_metrics functionality.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    benchmark : pl.Series, optional
        Benchmark returns for comparison metrics.
    risk_free_rate : float, optional
        Annualized risk-free rate.
    periods_per_year : int, optional
        Periods per year for annualization.
    var_confidence : float, optional
        Confidence level for VaR/CVaR.

    Returns
    -------
    MetricsReport
        Comprehensive metrics report.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02])
    >>> report = full_metrics(returns)
    >>> report.returns_metrics["total_return"]
    0.014...
    >>> report.to_json()
    '{"returns": {...}, "risk": {...}, ...}'
    """
    config = get_config()
    rf = risk_free_rate if risk_free_rate is not None else config.risk_free_rate
    ppy = periods_per_year or config.periods_per_year
    confidence = var_confidence or config.var_confidence

    returns_metrics = compute_returns_metrics(returns, periods_per_year=ppy)
    risk_metrics = compute_risk_metrics(
        returns, periods_per_year=ppy, var_confidence=confidence
    )
    performance_metrics = compute_performance_metrics(
        returns, risk_free_rate=rf, periods_per_year=ppy
    )
    distribution_metrics = compute_distribution_metrics(returns)
    trading_metrics = compute_trading_metrics(
        returns, risk_free_rate=rf, periods_per_year=ppy
    )

    benchmark_metrics = None
    if benchmark is not None:
        benchmark_metrics = compute_benchmark_metrics(
            returns, benchmark, risk_free_rate=rf, periods_per_year=ppy
        )

    return MetricsReport(
        returns_metrics=returns_metrics,
        risk_metrics=risk_metrics,
        performance_metrics=performance_metrics,
        distribution_metrics=distribution_metrics,
        trading_metrics=trading_metrics,
        benchmark_metrics=benchmark_metrics,
    )


def metrics_summary(
    returns: pl.Series,
    benchmark: pl.Series | None = None,
    *,
    risk_free_rate: float | None = None,
    periods_per_year: int | None = None,
) -> dict[str, Any]:
    """Compute key metrics summary.

    A lighter version of full_metrics that returns only the most
    commonly used metrics.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    benchmark : pl.Series, optional
        Benchmark returns for comparison.
    risk_free_rate : float, optional
        Annualized risk-free rate.
    periods_per_year : int, optional
        Periods per year for annualization.

    Returns
    -------
    dict
        Dictionary of key metrics.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02])
    >>> summary = metrics_summary(returns)
    >>> "sharpe" in summary
    True
    """
    config = get_config()
    rf = risk_free_rate if risk_free_rate is not None else config.risk_free_rate
    ppy = periods_per_year or config.periods_per_year

    result = {
        "total_return": comp(returns),
        "cagr": cagr(returns, periods_per_year=ppy),
        "volatility": volatility(returns, periods_per_year=ppy),
        "sharpe": sharpe(returns, risk_free_rate=rf, periods_per_year=ppy),
        "sortino": sortino(returns, risk_free_rate=rf, periods_per_year=ppy),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar(returns, periods_per_year=ppy),
        "win_rate": win_rate(returns),
        "best": best(returns),
        "worst": worst(returns),
    }

    if benchmark is not None:
        alpha, beta = greeks(returns, benchmark, periods_per_year=ppy)
        result["alpha"] = alpha
        result["beta"] = beta
        result["information_ratio"] = information_ratio(returns, benchmark)

    return result
