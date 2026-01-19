"""HTML report generation.

This module provides functions to generate HTML tearsheet reports
for strategy performance analysis.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from nanuquant.reports.metrics import full_metrics, MetricsReport


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        header {{
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1 {{
            color: #1a1a2e;
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: #666;
            font-size: 14px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 20px;
        }}
        .metric-card h3 {{
            color: #1a1a2e;
            font-size: 16px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }}
        .metric-row:last-child {{
            border-bottom: none;
        }}
        .metric-name {{
            color: #555;
            font-size: 14px;
        }}
        .metric-value {{
            font-weight: 600;
            color: #1a1a2e;
            font-size: 14px;
        }}
        .metric-value.positive {{
            color: #22c55e;
        }}
        .metric-value.negative {{
            color: #ef4444;
        }}
        footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #888;
            font-size: 12px;
            text-align: center;
        }}
        .highlight-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .highlight-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .highlight-value {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .highlight-label {{
            font-size: 12px;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="subtitle">Generated on {date} | {data_points} data points</p>
        </header>

        <div class="highlight-metrics">
            {highlight_cards}
        </div>

        <div class="metrics-grid">
            {metric_cards}
        </div>

        <footer>
            Generated with polars-metrics | Native Polars implementation
        </footer>
    </div>
</body>
</html>
"""


def _format_value(value: Any, name: str = "") -> tuple[str, str]:
    """Format a metric value for display.

    Returns (formatted_value, css_class).
    """
    if value is None:
        return "N/A", ""

    # Determine CSS class based on value and name
    css_class = ""
    name_lower = name.lower()

    # Percentage metrics
    pct_metrics = [
        "return",
        "cagr",
        "volatility",
        "win_rate",
        "exposure",
        "drawdown",
        "var",
        "cvar",
    ]

    if isinstance(value, (int, float)):
        # Determine if positive/negative color applies
        if any(m in name_lower for m in ["return", "alpha", "sharpe", "sortino", "calmar"]):
            css_class = "positive" if value > 0 else "negative" if value < 0 else ""
        elif "drawdown" in name_lower:
            css_class = "negative" if value < 0 else ""

        # Format as percentage or number
        if any(m in name_lower for m in pct_metrics):
            return f"{value * 100:.2f}%", css_class
        elif isinstance(value, int) or (isinstance(value, float) and value == int(value)):
            return f"{int(value):,}", css_class
        else:
            return f"{value:.4f}", css_class

    return str(value), css_class


def _create_metric_card(title: str, metrics: dict[str, Any]) -> str:
    """Create an HTML metric card."""
    rows = []
    for name, value in metrics.items():
        # Convert snake_case to Title Case
        display_name = name.replace("_", " ").title()
        formatted_value, css_class = _format_value(value, name)
        class_attr = f'class="metric-value {css_class}"' if css_class else 'class="metric-value"'
        rows.append(
            f'<div class="metric-row">'
            f'<span class="metric-name">{display_name}</span>'
            f'<span {class_attr}>{formatted_value}</span>'
            f"</div>"
        )

    return f"""
    <div class="metric-card">
        <h3>{title}</h3>
        {"".join(rows)}
    </div>
    """


def _create_highlight_card(label: str, value: Any, name: str = "") -> str:
    """Create an HTML highlight card."""
    formatted_value, _ = _format_value(value, name)
    return f"""
    <div class="highlight-card">
        <div class="highlight-value">{formatted_value}</div>
        <div class="highlight-label">{label}</div>
    </div>
    """


def generate_html_report(
    returns: pl.Series,
    benchmark: pl.Series | None = None,
    *,
    title: str = "Strategy Performance Report",
    risk_free_rate: float | None = None,
    periods_per_year: int | None = None,
) -> str:
    """Generate an HTML performance report.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    benchmark : pl.Series, optional
        Benchmark returns for comparison.
    title : str, default "Strategy Performance Report"
        Report title.
    risk_free_rate : float, optional
        Annualized risk-free rate.
    periods_per_year : int, optional
        Periods per year for annualization.

    Returns
    -------
    str
        HTML report as a string.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02])
    >>> html = generate_html_report(returns, title="My Strategy")
    >>> "My Strategy" in html
    True
    """
    report = full_metrics(
        returns,
        benchmark,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    # Create highlight cards for key metrics
    highlight_cards = []
    highlight_cards.append(
        _create_highlight_card("Total Return", report.returns_metrics["total_return"], "total_return")
    )
    highlight_cards.append(
        _create_highlight_card("Sharpe Ratio", report.performance_metrics["sharpe"], "sharpe")
    )
    highlight_cards.append(
        _create_highlight_card("Max Drawdown", report.risk_metrics["max_drawdown"], "max_drawdown")
    )
    highlight_cards.append(
        _create_highlight_card("Win Rate", report.returns_metrics["win_rate"], "win_rate")
    )
    if report.benchmark_metrics:
        highlight_cards.append(
            _create_highlight_card("Alpha", report.benchmark_metrics["alpha"], "alpha")
        )

    # Create metric cards
    metric_cards = []
    metric_cards.append(_create_metric_card("Returns", report.returns_metrics))
    metric_cards.append(_create_metric_card("Risk", report.risk_metrics))
    metric_cards.append(_create_metric_card("Performance", report.performance_metrics))
    metric_cards.append(_create_metric_card("Distribution", report.distribution_metrics))
    metric_cards.append(_create_metric_card("Trading", report.trading_metrics))
    if report.benchmark_metrics:
        metric_cards.append(_create_metric_card("Benchmark Comparison", report.benchmark_metrics))

    html = HTML_TEMPLATE.format(
        title=title,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        data_points=len(returns),
        highlight_cards="".join(highlight_cards),
        metric_cards="".join(metric_cards),
    )

    return html


def save_html_report(
    returns: pl.Series,
    filepath: str | Path,
    benchmark: pl.Series | None = None,
    *,
    title: str = "Strategy Performance Report",
    risk_free_rate: float | None = None,
    periods_per_year: int | None = None,
) -> Path:
    """Generate and save an HTML performance report.

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    filepath : str or Path
        Output file path.
    benchmark : pl.Series, optional
        Benchmark returns for comparison.
    title : str, default "Strategy Performance Report"
        Report title.
    risk_free_rate : float, optional
        Annualized risk-free rate.
    periods_per_year : int, optional
        Periods per year for annualization.

    Returns
    -------
    Path
        Path to the saved report.

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02])
    >>> path = save_html_report(returns, "report.html")
    >>> path.exists()
    True
    """
    html = generate_html_report(
        returns,
        benchmark,
        title=title,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    filepath = Path(filepath)
    filepath.write_text(html)
    return filepath
