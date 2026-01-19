"""Polars Expression Namespace plugin for nanuquant.

This module provides a Polars namespace plugin that allows idiomatic usage
of metrics directly on Polars expressions.

Examples
--------
>>> import polars as pl
>>> import nanuquant as pm
>>>
>>> df = pl.DataFrame({"returns": [0.01, -0.02, 0.015, -0.01, 0.02] * 50})
>>> df.select(pl.col("returns").metrics.sharpe())
>>> df.select(pl.col("returns").metrics.volatility(periods_per_year=252))
>>> df.with_columns(pl.col("returns").metrics.rolling_sharpe().alias("rolling_sharpe"))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr


@pl.api.register_expr_namespace("metrics")
class MetricsNamespace:
    """Polars expression namespace for quantitative finance metrics.

    This namespace provides access to nanuquant functions directly on
    Polars expressions, enabling idiomatic usage in select, with_columns,
    and group_by contexts.

    Examples
    --------
    >>> import polars as pl
    >>> import nanuquant as pm  # Registers the namespace
    >>>
    >>> df = pl.DataFrame({"returns": [0.01, -0.02, 0.015, -0.01, 0.02] * 50})
    >>>
    >>> # Single metric
    >>> df.select(pl.col("returns").metrics.sharpe())
    >>>
    >>> # Multiple metrics
    >>> df.select([
    ...     pl.col("returns").metrics.sharpe().alias("sharpe"),
    ...     pl.col("returns").metrics.sortino().alias("sortino"),
    ...     pl.col("returns").metrics.max_drawdown().alias("max_dd"),
    ... ])
    >>>
    >>> # Rolling metrics in with_columns
    >>> df.with_columns([
    ...     pl.col("returns").metrics.rolling_volatility().alias("rolling_vol"),
    ... ])
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def _apply_metric(
        self,
        func: callable,
        return_dtype: pl.DataType = pl.Float64,
        **kwargs: object,
    ) -> pl.Expr:
        """Apply a metric function to the expression using map_batches."""
        def apply_func(s: pl.Series) -> pl.Series:
            result = func(s, **kwargs)
            if isinstance(result, pl.Series):
                return result
            # Scalar result - return as single-element series
            return pl.Series([result], dtype=return_dtype)

        return self._expr.map_batches(apply_func, return_dtype=return_dtype)

    # =========================================================================
    # Return Metrics
    # =========================================================================

    def comp(self) -> pl.Expr:
        """Calculate total compounded return."""
        from nanuquant.core import comp

        return self._apply_metric(comp)

    def cagr(self, *, periods_per_year: int = 252) -> pl.Expr:
        """Calculate Compound Annual Growth Rate."""
        from nanuquant.core import cagr

        return self._apply_metric(cagr, periods_per_year=periods_per_year)

    def avg_return(self) -> pl.Expr:
        """Calculate average return."""
        from nanuquant.core import avg_return

        return self._apply_metric(avg_return)

    def avg_win(self) -> pl.Expr:
        """Calculate average winning return."""
        from nanuquant.core import avg_win

        return self._apply_metric(avg_win)

    def avg_loss(self) -> pl.Expr:
        """Calculate average losing return."""
        from nanuquant.core import avg_loss

        return self._apply_metric(avg_loss)

    def best(self) -> pl.Expr:
        """Get best (maximum) return."""
        from nanuquant.core import best

        return self._apply_metric(best)

    def worst(self) -> pl.Expr:
        """Get worst (minimum) return."""
        from nanuquant.core import worst

        return self._apply_metric(worst)

    def win_rate(self) -> pl.Expr:
        """Calculate win rate (fraction of positive returns)."""
        from nanuquant.core import win_rate

        return self._apply_metric(win_rate)

    def payoff_ratio(self) -> pl.Expr:
        """Calculate payoff ratio (avg win / avg loss)."""
        from nanuquant.core import payoff_ratio

        return self._apply_metric(payoff_ratio)

    def profit_factor(self) -> pl.Expr:
        """Calculate profit factor (sum wins / sum losses)."""
        from nanuquant.core import profit_factor

        return self._apply_metric(profit_factor)

    # =========================================================================
    # Risk Metrics
    # =========================================================================

    def volatility(self, *, periods_per_year: int = 252) -> pl.Expr:
        """Calculate annualized volatility."""
        from nanuquant.core import volatility

        return self._apply_metric(volatility, periods_per_year=periods_per_year)

    def var(self, *, confidence: float = 0.95) -> pl.Expr:
        """Calculate Value at Risk."""
        from nanuquant.core import var

        return self._apply_metric(var, confidence=confidence)

    def cvar(self, *, confidence: float = 0.95) -> pl.Expr:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        from nanuquant.core import cvar

        return self._apply_metric(cvar, confidence=confidence)

    def max_drawdown(self) -> pl.Expr:
        """Calculate maximum drawdown."""
        from nanuquant.core import max_drawdown

        return self._apply_metric(max_drawdown)

    def ulcer_index(self) -> pl.Expr:
        """Calculate Ulcer Index."""
        from nanuquant.core import ulcer_index

        return self._apply_metric(ulcer_index)

    def downside_deviation(
        self, *, mar: float = 0.0, periods_per_year: int = 252
    ) -> pl.Expr:
        """Calculate downside deviation."""
        from nanuquant.core import downside_deviation

        return self._apply_metric(
            downside_deviation, mar=mar, periods_per_year=periods_per_year
        )

    # =========================================================================
    # Performance Metrics
    # =========================================================================

    def sharpe(
        self, *, risk_free_rate: float = 0.0, periods_per_year: int = 252
    ) -> pl.Expr:
        """Calculate Sharpe ratio."""
        from nanuquant.core import sharpe

        return self._apply_metric(
            sharpe, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        )

    def sortino(
        self, *, risk_free_rate: float = 0.0, periods_per_year: int = 252
    ) -> pl.Expr:
        """Calculate Sortino ratio."""
        from nanuquant.core import sortino

        return self._apply_metric(
            sortino, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        )

    def calmar(self, *, periods_per_year: int = 252) -> pl.Expr:
        """Calculate Calmar ratio."""
        from nanuquant.core import calmar

        return self._apply_metric(calmar, periods_per_year=periods_per_year)

    def omega(
        self,
        *,
        threshold: float = 0.0,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> pl.Expr:
        """Calculate Omega ratio."""
        from nanuquant.core import omega

        return self._apply_metric(
            omega,
            threshold=threshold,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

    def gain_to_pain_ratio(self) -> pl.Expr:
        """Calculate Gain to Pain ratio."""
        from nanuquant.core import gain_to_pain_ratio

        return self._apply_metric(gain_to_pain_ratio)

    def ulcer_performance_index(self, *, risk_free_rate: float = 0.0) -> pl.Expr:
        """Calculate Ulcer Performance Index."""
        from nanuquant.core import ulcer_performance_index

        return self._apply_metric(
            ulcer_performance_index, risk_free_rate=risk_free_rate
        )

    def kelly_criterion(self) -> pl.Expr:
        """Calculate Kelly Criterion."""
        from nanuquant.core import kelly_criterion

        return self._apply_metric(kelly_criterion)

    def tail_ratio(self, *, confidence: float = 0.95) -> pl.Expr:
        """Calculate Tail ratio."""
        from nanuquant.core import tail_ratio

        return self._apply_metric(tail_ratio, confidence=confidence)

    def common_sense_ratio(self) -> pl.Expr:
        """Calculate Common Sense ratio."""
        from nanuquant.core import common_sense_ratio

        return self._apply_metric(common_sense_ratio)

    def recovery_factor(self) -> pl.Expr:
        """Calculate Recovery Factor."""
        from nanuquant.core import recovery_factor

        return self._apply_metric(recovery_factor)

    def risk_return_ratio(self) -> pl.Expr:
        """Calculate Risk/Return ratio."""
        from nanuquant.core import risk_return_ratio

        return self._apply_metric(risk_return_ratio)

    # =========================================================================
    # Distribution Metrics
    # =========================================================================

    def skewness(self) -> pl.Expr:
        """Calculate skewness."""
        from nanuquant.core import skewness

        return self._apply_metric(skewness)

    def kurtosis(self, *, excess: bool = True) -> pl.Expr:
        """Calculate kurtosis."""
        from nanuquant.core import kurtosis

        return self._apply_metric(kurtosis, excess=excess)

    def expected_return(self, *, periods_per_year: int = 252) -> pl.Expr:
        """Calculate expected (annualized) return."""
        from nanuquant.core import expected_return

        return self._apply_metric(expected_return, periods_per_year=periods_per_year)

    def geometric_mean(self) -> pl.Expr:
        """Calculate geometric mean return."""
        from nanuquant.core import geometric_mean

        return self._apply_metric(geometric_mean)

    # =========================================================================
    # Rolling Metrics (return Series, not scalars)
    # =========================================================================

    def rolling_volatility(
        self,
        *,
        rolling_period: int = 126,
        periods_per_year: int | None = None,
        annualize: bool = True,
    ) -> pl.Expr:
        """Calculate rolling volatility.

        Parameters
        ----------
        rolling_period : int, default 126
            Window size for rolling calculation.
        periods_per_year : int, optional
            Periods per year for annualization. If None, uses 365.
        annualize : bool, default True
            If True, annualize the volatility.

        Returns
        -------
        pl.Expr
            Expression returning rolling volatility series.
        """
        from nanuquant.core import rolling_volatility

        return self._apply_metric(
            rolling_volatility,
            rolling_period=rolling_period,
            periods_per_year=periods_per_year,
            annualize=annualize,
        )

    def rolling_sharpe(
        self,
        *,
        risk_free_rate: float = 0.0,
        rolling_period: int = 126,
        periods_per_year: int | None = None,
        annualize: bool = True,
    ) -> pl.Expr:
        """Calculate rolling Sharpe ratio.

        Parameters
        ----------
        risk_free_rate : float, default 0.0
            Annualized risk-free rate.
        rolling_period : int, default 126
            Window size for rolling calculation.
        periods_per_year : int, optional
            Periods per year for annualization. If None, uses 365.
        annualize : bool, default True
            If True, annualize the ratio.

        Returns
        -------
        pl.Expr
            Expression returning rolling Sharpe series.
        """
        from nanuquant.core import rolling_sharpe

        return self._apply_metric(
            rolling_sharpe,
            risk_free_rate=risk_free_rate,
            rolling_period=rolling_period,
            periods_per_year=periods_per_year,
            annualize=annualize,
        )

    def rolling_sortino(
        self,
        *,
        risk_free_rate: float = 0.0,
        rolling_period: int = 126,
        periods_per_year: int | None = None,
        annualize: bool = True,
    ) -> pl.Expr:
        """Calculate rolling Sortino ratio.

        Parameters
        ----------
        risk_free_rate : float, default 0.0
            Annualized risk-free rate.
        rolling_period : int, default 126
            Window size for rolling calculation.
        periods_per_year : int, optional
            Periods per year for annualization. If None, uses 365.
        annualize : bool, default True
            If True, annualize the ratio.

        Returns
        -------
        pl.Expr
            Expression returning rolling Sortino series.
        """
        from nanuquant.core import rolling_sortino

        return self._apply_metric(
            rolling_sortino,
            risk_free_rate=risk_free_rate,
            rolling_period=rolling_period,
            periods_per_year=periods_per_year,
            annualize=annualize,
        )

    # =========================================================================
    # Drawdown Series (returns Series)
    # =========================================================================

    def to_drawdown_series(self) -> pl.Expr:
        """Calculate drawdown series.

        Returns
        -------
        pl.Expr
            Expression returning drawdown series (negative values representing drawdowns).
        """
        from nanuquant.core import to_drawdown_series

        return self._apply_metric(to_drawdown_series)

    # =========================================================================
    # Streak Metrics
    # =========================================================================

    def consecutive_wins(self) -> pl.Expr:
        """Calculate maximum consecutive winning periods."""
        from nanuquant.core import consecutive_wins

        return self._apply_metric(consecutive_wins, return_dtype=pl.Int64)

    def consecutive_losses(self) -> pl.Expr:
        """Calculate maximum consecutive losing periods."""
        from nanuquant.core import consecutive_losses

        return self._apply_metric(consecutive_losses, return_dtype=pl.Int64)
