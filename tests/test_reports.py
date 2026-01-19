"""Tests for the reports module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

from nanuquant.reports import (
    MetricsReport,
    compute_benchmark_metrics,
    compute_distribution_metrics,
    compute_performance_metrics,
    compute_returns_metrics,
    compute_risk_metrics,
    compute_trading_metrics,
    full_metrics,
    generate_html_report,
    metrics_summary,
    save_html_report,
)


@pytest.fixture
def sample_returns() -> pl.Series:
    """Create sample returns for testing."""
    return pl.Series(
        "returns",
        [0.01, -0.02, 0.015, -0.01, 0.02, 0.005, -0.008, 0.012, -0.003, 0.018],
    )


@pytest.fixture
def benchmark_returns() -> pl.Series:
    """Create benchmark returns for testing."""
    return pl.Series(
        "benchmark",
        [0.008, -0.015, 0.01, -0.008, 0.015, 0.003, -0.005, 0.009, -0.002, 0.012],
    )


class TestComputeMetrics:
    """Test individual compute functions."""

    def test_compute_returns_metrics(self, sample_returns: pl.Series) -> None:
        """Test compute_returns_metrics returns expected keys."""
        metrics = compute_returns_metrics(sample_returns)

        assert "total_return" in metrics
        assert "cagr" in metrics
        assert "avg_return" in metrics
        assert "avg_win" in metrics
        assert "avg_loss" in metrics
        assert "best" in metrics
        assert "worst" in metrics
        assert "win_rate" in metrics
        assert "payoff_ratio" in metrics
        assert "profit_factor" in metrics
        assert "consecutive_wins" in metrics
        assert "consecutive_losses" in metrics

        # Verify reasonable values
        assert isinstance(metrics["total_return"], float)
        assert 0 <= metrics["win_rate"] <= 1
        assert metrics["best"] >= metrics["worst"]

    def test_compute_risk_metrics(self, sample_returns: pl.Series) -> None:
        """Test compute_risk_metrics returns expected keys."""
        metrics = compute_risk_metrics(sample_returns)

        assert "volatility" in metrics
        assert "var" in metrics
        assert "cvar" in metrics
        assert "max_drawdown" in metrics
        assert "ulcer_index" in metrics
        assert "downside_deviation" in metrics

        # Verify reasonable values
        assert metrics["volatility"] > 0
        assert metrics["var"] < 0  # VaR is a loss
        assert metrics["max_drawdown"] <= 0  # Drawdown is negative

    def test_compute_performance_metrics(self, sample_returns: pl.Series) -> None:
        """Test compute_performance_metrics returns expected keys."""
        metrics = compute_performance_metrics(sample_returns)

        assert "sharpe" in metrics
        assert "sortino" in metrics
        assert "calmar" in metrics
        assert "omega" in metrics
        assert "gain_to_pain_ratio" in metrics
        assert "ulcer_performance_index" in metrics
        assert "kelly_criterion" in metrics
        assert "tail_ratio" in metrics
        assert "common_sense_ratio" in metrics
        assert "risk_return_ratio" in metrics
        assert "recovery_factor" in metrics

    def test_compute_distribution_metrics(self, sample_returns: pl.Series) -> None:
        """Test compute_distribution_metrics returns expected keys."""
        metrics = compute_distribution_metrics(sample_returns)

        assert "skewness" in metrics
        assert "kurtosis" in metrics
        assert "jarque_bera_stat" in metrics
        assert "jarque_bera_pvalue" in metrics
        assert "shapiro_wilk_stat" in metrics
        assert "shapiro_wilk_pvalue" in metrics
        assert "outlier_win_ratio" in metrics
        assert "outlier_loss_ratio" in metrics
        assert "expected_return" in metrics
        assert "geometric_mean" in metrics

    def test_compute_trading_metrics(self, sample_returns: pl.Series) -> None:
        """Test compute_trading_metrics returns expected keys."""
        metrics = compute_trading_metrics(sample_returns)

        assert "exposure" in metrics
        assert "ghpr" in metrics
        assert "rar" in metrics
        assert "cpc_index" in metrics
        assert "serenity_index" in metrics
        assert "risk_of_ruin" in metrics
        assert "adjusted_sortino" in metrics
        assert "smart_sharpe" in metrics
        assert "smart_sortino" in metrics
        assert "sqn" in metrics
        assert "expectancy" in metrics
        assert "k_ratio" in metrics

    def test_compute_benchmark_metrics(
        self, sample_returns: pl.Series, benchmark_returns: pl.Series
    ) -> None:
        """Test compute_benchmark_metrics returns expected keys."""
        metrics = compute_benchmark_metrics(sample_returns, benchmark_returns)

        assert "alpha" in metrics
        assert "beta" in metrics
        assert "information_ratio" in metrics
        assert "r_squared" in metrics
        assert "treynor_ratio" in metrics


class TestFullMetrics:
    """Test full_metrics function."""

    def test_full_metrics_returns_report(self, sample_returns: pl.Series) -> None:
        """Test full_metrics returns MetricsReport."""
        report = full_metrics(sample_returns)

        assert isinstance(report, MetricsReport)
        assert report.returns_metrics is not None
        assert report.risk_metrics is not None
        assert report.performance_metrics is not None
        assert report.distribution_metrics is not None
        assert report.trading_metrics is not None
        assert report.benchmark_metrics is None

    def test_full_metrics_with_benchmark(
        self, sample_returns: pl.Series, benchmark_returns: pl.Series
    ) -> None:
        """Test full_metrics with benchmark."""
        report = full_metrics(sample_returns, benchmark_returns)

        assert report.benchmark_metrics is not None
        assert "alpha" in report.benchmark_metrics
        assert "beta" in report.benchmark_metrics

    def test_full_metrics_custom_params(self, sample_returns: pl.Series) -> None:
        """Test full_metrics with custom parameters."""
        report = full_metrics(
            sample_returns,
            risk_free_rate=0.02,
            periods_per_year=252,  # Keep same periods for fair comparison
            var_confidence=0.99,
        )

        # With higher rf rate (same periods_per_year), Sharpe should be lower
        report_default = full_metrics(sample_returns, risk_free_rate=0.0, periods_per_year=252)
        assert report.performance_metrics["sharpe"] < report_default.performance_metrics["sharpe"]


class TestMetricsReport:
    """Test MetricsReport class methods."""

    def test_to_dict(self, sample_returns: pl.Series) -> None:
        """Test to_dict method."""
        report = full_metrics(sample_returns)
        d = report.to_dict()

        assert isinstance(d, dict)
        assert "returns" in d
        assert "risk" in d
        assert "performance" in d
        assert "distribution" in d
        assert "trading" in d

    def test_to_dict_with_benchmark(
        self, sample_returns: pl.Series, benchmark_returns: pl.Series
    ) -> None:
        """Test to_dict includes benchmark when present."""
        report = full_metrics(sample_returns, benchmark_returns)
        d = report.to_dict()

        assert "benchmark" in d

    def test_to_json(self, sample_returns: pl.Series) -> None:
        """Test to_json method."""
        report = full_metrics(sample_returns)
        json_str = report.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "returns" in parsed
        assert "risk" in parsed

    def test_to_json_with_indent(self, sample_returns: pl.Series) -> None:
        """Test to_json with custom indent."""
        report = full_metrics(sample_returns)
        json_compact = report.to_json(indent=None)
        json_formatted = report.to_json(indent=4)

        # Formatted should be longer due to whitespace
        assert len(json_formatted) > len(json_compact)

    def test_to_polars(self, sample_returns: pl.Series) -> None:
        """Test to_polars method."""
        report = full_metrics(sample_returns)
        df = report.to_polars()

        assert isinstance(df, pl.DataFrame)
        assert "category" in df.columns
        assert "metric" in df.columns
        assert "value" in df.columns
        assert len(df) > 0


class TestMetricsSummary:
    """Test metrics_summary function."""

    def test_metrics_summary_returns_dict(self, sample_returns: pl.Series) -> None:
        """Test metrics_summary returns dictionary."""
        summary = metrics_summary(sample_returns)

        assert isinstance(summary, dict)
        assert "total_return" in summary
        assert "cagr" in summary
        assert "volatility" in summary
        assert "sharpe" in summary
        assert "sortino" in summary
        assert "max_drawdown" in summary
        assert "calmar" in summary
        assert "win_rate" in summary
        assert "best" in summary
        assert "worst" in summary

    def test_metrics_summary_with_benchmark(
        self, sample_returns: pl.Series, benchmark_returns: pl.Series
    ) -> None:
        """Test metrics_summary with benchmark."""
        summary = metrics_summary(sample_returns, benchmark_returns)

        assert "alpha" in summary
        assert "beta" in summary
        assert "information_ratio" in summary

    def test_metrics_summary_fewer_keys_than_full(
        self, sample_returns: pl.Series
    ) -> None:
        """Test metrics_summary has fewer keys than full_metrics."""
        summary = metrics_summary(sample_returns)
        report = full_metrics(sample_returns)

        total_full = sum(len(v) for v in report.to_dict().values())
        assert len(summary) < total_full


class TestHTMLReport:
    """Test HTML report generation."""

    def test_generate_html_report(self, sample_returns: pl.Series) -> None:
        """Test generate_html_report returns valid HTML."""
        html = generate_html_report(sample_returns)

        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

    def test_generate_html_report_custom_title(
        self, sample_returns: pl.Series
    ) -> None:
        """Test generate_html_report with custom title."""
        title = "My Custom Strategy Report"
        html = generate_html_report(sample_returns, title=title)

        assert title in html

    def test_generate_html_report_with_benchmark(
        self, sample_returns: pl.Series, benchmark_returns: pl.Series
    ) -> None:
        """Test generate_html_report with benchmark."""
        html = generate_html_report(sample_returns, benchmark_returns)

        # Should include benchmark comparison section
        assert "Benchmark Comparison" in html
        assert "Alpha" in html
        assert "Beta" in html

    def test_generate_html_report_contains_metrics(
        self, sample_returns: pl.Series
    ) -> None:
        """Test generate_html_report contains metric sections."""
        html = generate_html_report(sample_returns)

        assert "Returns" in html
        assert "Risk" in html
        assert "Performance" in html
        assert "Distribution" in html
        assert "Trading" in html

    def test_save_html_report(self, sample_returns: pl.Series) -> None:
        """Test save_html_report creates file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.html"
            result = save_html_report(sample_returns, filepath)

            assert result == filepath
            assert filepath.exists()
            assert filepath.stat().st_size > 0

            # Verify content
            content = filepath.read_text()
            assert "<!DOCTYPE html>" in content

    def test_save_html_report_with_options(
        self, sample_returns: pl.Series, benchmark_returns: pl.Series
    ) -> None:
        """Test save_html_report with all options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "custom_report.html"
            result = save_html_report(
                sample_returns,
                filepath,
                benchmark=benchmark_returns,
                title="Custom Title",
                risk_free_rate=0.03,
                periods_per_year=252,
            )

            assert result == filepath
            content = filepath.read_text()
            assert "Custom Title" in content
            assert "Benchmark Comparison" in content


class TestEdgeCases:
    """Test edge cases for reports."""

    def test_short_series(self) -> None:
        """Test with very short series."""
        short = pl.Series("returns", [0.01, -0.01])
        report = full_metrics(short)

        assert report is not None
        # Some metrics may be NaN/None for short series
        assert report.returns_metrics["total_return"] is not None

    def test_all_positive_returns(self) -> None:
        """Test with all positive returns."""
        positive = pl.Series("returns", [0.01, 0.02, 0.015, 0.005, 0.01])
        report = full_metrics(positive)

        assert report.returns_metrics["win_rate"] == 1.0
        assert report.returns_metrics["avg_loss"] is None or report.returns_metrics["avg_loss"] == 0

    def test_all_negative_returns(self) -> None:
        """Test with all negative returns."""
        import math

        negative = pl.Series("returns", [-0.01, -0.02, -0.015, -0.005, -0.01])
        report = full_metrics(negative)

        assert report.returns_metrics["win_rate"] == 0.0
        # avg_win may be None, 0, or NaN for all-negative returns
        avg_win = report.returns_metrics["avg_win"]
        assert avg_win is None or avg_win == 0 or (isinstance(avg_win, float) and math.isnan(avg_win))
