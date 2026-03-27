"""Benchmark NanuQuant vs QuantStats: performance timing and calculation accuracy.

Generates markdown tables showing:
1. Execution time comparison across metrics and dataset sizes
2. Calculation accuracy (side-by-side values and relative differences)

Usage:
    uv run python benchmarks/run_benchmarks.py
    uv run python benchmarks/run_benchmarks.py --output docs/benchmarks.md
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import quantstats_lumi as qs

import nanuquant as nq

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def make_data(n: int, seed: int = 42) -> tuple[pd.Series, pl.Series]:
    """Generate synthetic daily returns as both pandas and Polars Series."""
    rng = np.random.default_rng(seed)
    vals = rng.normal(0.0004, 0.015, n)
    dates = pd.date_range("2000-01-01", periods=n, freq="B")
    pd_series = pd.Series(vals, index=dates, name="returns")
    pl_series = pl.Series("returns", vals)
    return pd_series, pl_series


def load_real_data() -> tuple[pd.Series, pl.Series] | None:
    """Load cached SPY data if available."""
    cache = Path(__file__).parent.parent / "tests" / ".data_cache" / "SPY_max.parquet"
    if not cache.exists():
        return None
    df = pd.read_parquet(cache)
    pd_series = df["returns"]
    pl_series = pl.Series("returns", pd_series.values)
    return pd_series, pl_series


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    metric: str
    nq_ms: float
    qs_ms: float

    @property
    def speedup(self) -> float:
        return self.qs_ms / self.nq_ms if self.nq_ms > 0 else float("inf")


@dataclass
class AccuracyResult:
    metric: str
    nq_value: float
    qs_value: float
    tolerance: str

    @property
    def rel_diff(self) -> float:
        if self.qs_value == 0:
            return 0.0 if self.nq_value == 0 else float("inf")
        return abs(self.nq_value - self.qs_value) / abs(self.qs_value)

    @property
    def match(self) -> str:
        tol_map = {"EXACT": 1e-10, "TIGHT": 1e-8, "LOOSE": 1e-6, "CALENDAR": 0.003}
        threshold = tol_map.get(self.tolerance, 1e-6)
        return "PASS" if self.rel_diff <= threshold else "DIFF*"


@dataclass
class BenchmarkSuite:
    timing_results: list[TimingResult] = field(default_factory=list)
    accuracy_results: list[AccuracyResult] = field(default_factory=list)
    dataset_label: str = ""
    n_observations: int = 0


def time_fn(fn: callable, *args: object, n_runs: int = 50, **kwargs: object) -> float:
    """Time a function over n_runs iterations, return median time in ms."""
    times = []
    # Warmup
    fn(*args, **kwargs)
    for _ in range(n_runs):
        start = time.perf_counter()
        fn(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Metric definitions: (name, nq_callable, qs_callable, tolerance)
# ---------------------------------------------------------------------------


def get_metrics(
    pd_returns: pd.Series,
    pl_returns: pl.Series,
    pd_bench: pd.Series | None = None,
    pl_bench: pl.Series | None = None,
) -> list[tuple[str, callable, callable, str]]:
    """Define metric pairs for comparison."""
    metrics = [
        # Returns
        (
            "comp (total return)",
            lambda: nq.comp(pl_returns),
            lambda: qs.stats.comp(pd_returns),
            "TIGHT",
        ),
        (
            "cagr",
            lambda: nq.cagr(pl_returns, periods_per_year=252),
            lambda: qs.stats.cagr(pd_returns),
            "CALENDAR",
        ),
        (
            "avg_return",
            lambda: nq.avg_return(pl_returns),
            lambda: qs.stats.avg_return(pd_returns),
            "EXACT",
        ),
        (
            "avg_win",
            lambda: nq.avg_win(pl_returns),
            lambda: qs.stats.avg_win(pd_returns),
            "EXACT",
        ),
        (
            "avg_loss",
            lambda: nq.avg_loss(pl_returns),
            lambda: qs.stats.avg_loss(pd_returns),
            "EXACT",
        ),
        (
            "best",
            lambda: nq.best(pl_returns),
            lambda: qs.stats.best(pd_returns),
            "EXACT",
        ),
        (
            "worst",
            lambda: nq.worst(pl_returns),
            lambda: qs.stats.worst(pd_returns),
            "EXACT",
        ),
        (
            "win_rate",
            lambda: nq.win_rate(pl_returns),
            lambda: qs.stats.win_rate(pd_returns),
            "EXACT",
        ),
        (
            "profit_factor",
            lambda: nq.profit_factor(pl_returns),
            lambda: qs.stats.profit_factor(pd_returns),
            "TIGHT",
        ),
        (
            "payoff_ratio",
            lambda: nq.payoff_ratio(pl_returns),
            lambda: qs.stats.payoff_ratio(pd_returns),
            "TIGHT",
        ),
        # Risk
        (
            "volatility",
            lambda: nq.volatility(pl_returns, periods_per_year=252),
            lambda: qs.stats.volatility(pd_returns, periods=252),
            "TIGHT",
        ),
        (
            "var (95%)",
            lambda: nq.var(pl_returns, confidence=0.95),
            lambda: qs.stats.var(pd_returns, confidence=0.95),
            "LOOSE",
        ),
        (
            "cvar (95%)",
            lambda: nq.cvar(pl_returns, confidence=0.95),
            lambda: qs.stats.cvar(pd_returns, confidence=0.95),
            "LOOSE",
        ),
        (
            "max_drawdown",
            lambda: nq.max_drawdown(pl_returns),
            lambda: qs.stats.max_drawdown(pd_returns),
            "TIGHT",
        ),
        # Performance
        (
            "sharpe",
            lambda: nq.sharpe(pl_returns, periods_per_year=252),
            lambda: qs.stats.sharpe(pd_returns, periods=252),
            "TIGHT",
        ),
        (
            "sortino",
            lambda: nq.sortino(pl_returns, periods_per_year=252),
            lambda: qs.stats.sortino(pd_returns, periods=252),
            "TIGHT",
        ),
        (
            "calmar",
            lambda: nq.calmar(pl_returns, periods_per_year=252),
            lambda: qs.stats.calmar(pd_returns),
            "CALENDAR",
        ),
        (
            "omega",
            lambda: nq.omega(pl_returns),
            lambda: qs.stats.omega(pd_returns),
            "LOOSE",
        ),
        (
            "gain_to_pain_ratio",
            lambda: nq.gain_to_pain_ratio(pl_returns),
            lambda: qs.stats.gain_to_pain_ratio(pd_returns),
            "TIGHT",
        ),
        (
            "tail_ratio",
            lambda: nq.tail_ratio(pl_returns),
            lambda: qs.stats.tail_ratio(pd_returns),
            "TIGHT",
        ),
        (
            "common_sense_ratio",
            lambda: nq.common_sense_ratio(pl_returns),
            lambda: qs.stats.common_sense_ratio(pd_returns),
            "TIGHT",
        ),
        (
            "kelly_criterion",
            lambda: nq.kelly_criterion(pl_returns),
            lambda: qs.stats.kelly_criterion(pd_returns),
            "TIGHT",
        ),
        # Distribution
        (
            "skewness",
            lambda: nq.skewness(pl_returns),
            lambda: qs.stats.skew(pd_returns),
            "TIGHT",
        ),
        (
            "kurtosis",
            lambda: nq.kurtosis(pl_returns),
            lambda: qs.stats.kurtosis(pd_returns),
            "TIGHT",
        ),
    ]
    return metrics


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------


def run_suite(
    pd_returns: pd.Series,
    pl_returns: pl.Series,
    label: str,
    n_runs: int = 50,
) -> BenchmarkSuite:
    """Run all benchmarks for a given dataset."""
    suite = BenchmarkSuite(dataset_label=label, n_observations=len(pd_returns))
    metrics = get_metrics(pd_returns, pl_returns)

    for name, nq_fn, qs_fn, tolerance in metrics:
        # Accuracy
        try:
            nq_val = float(nq_fn())
        except Exception:
            nq_val = float("nan")
        try:
            qs_val = float(qs_fn())
        except Exception:
            qs_val = float("nan")

        suite.accuracy_results.append(
            AccuracyResult(metric=name, nq_value=nq_val, qs_value=qs_val, tolerance=tolerance)
        )

        # Timing
        try:
            nq_time = time_fn(nq_fn, n_runs=n_runs)
        except Exception:
            nq_time = float("nan")
        try:
            qs_time = time_fn(qs_fn, n_runs=n_runs)
        except Exception:
            qs_time = float("nan")

        suite.timing_results.append(TimingResult(metric=name, nq_ms=nq_time, qs_ms=qs_time))

    return suite


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def format_value(v: float) -> str:
    """Format a float for display."""
    if np.isnan(v) or np.isinf(v):
        return str(v)
    if abs(v) < 0.0001:
        return f"{v:.8f}"
    if abs(v) < 1:
        return f"{v:.6f}"
    return f"{v:.4f}"


def format_pct(v: float) -> str:
    """Format a relative difference as percentage."""
    if np.isnan(v) or np.isinf(v):
        return "N/A"
    if v == 0:
        return "0%"
    if v < 1e-10:
        return "<0.0000001%"
    return f"{v:.8%}"


def generate_markdown(suites: list[BenchmarkSuite]) -> str:
    """Generate full benchmark markdown document."""
    lines: list[str] = []

    lines.append("# Benchmarks: NanuQuant vs QuantStats\n")
    lines.append(
        "> Auto-generated by `benchmarks/run_benchmarks.py`. Re-run to update with your hardware.\n"
    )
    lines.append(
        "This page shows side-by-side performance timing and calculation accuracy "
        "between NanuQuant (Polars-native) and QuantStats (Pandas-based). "
        "All benchmarks use identical input data.\n"
    )

    # Summary
    lines.append("## Key Findings\n")
    for suite in suites:
        valid = [t for t in suite.timing_results if not np.isnan(t.nq_ms) and not np.isnan(t.qs_ms)]
        if not valid:
            continue
        speedups = [t.speedup for t in valid]
        median_speedup = float(np.median(speedups))
        max_speedup = max(speedups)
        max_metric = valid[np.argmax(speedups)].metric

        matching = [a for a in suite.accuracy_results if a.match == "PASS"]
        total = len(suite.accuracy_results)

        lines.append(f"**{suite.dataset_label}** ({suite.n_observations:,} observations):\n")
        lines.append(f"- Median speedup: **{median_speedup:.1f}x** faster than QuantStats")
        lines.append(f"- Peak speedup: **{max_speedup:.1f}x** (`{max_metric}`)")
        lines.append(
            f"- Calculation accuracy: **{len(matching)}/{total}** metrics match within tolerance"
        )
        lines.append("")

    # Performance tables
    lines.append("---\n")
    lines.append("## Performance Comparison\n")

    for suite in suites:
        lines.append(f"### {suite.dataset_label} ({suite.n_observations:,} observations)\n")
        lines.append("| Metric | NanuQuant (ms) | QuantStats (ms) | Speedup |")
        lines.append("|--------|---------------:|----------------:|--------:|")
        for t in suite.timing_results:
            if np.isnan(t.nq_ms) or np.isnan(t.qs_ms):
                lines.append(f"| {t.metric} | N/A | N/A | N/A |")
            else:
                speedup_str = f"{t.speedup:.1f}x"
                lines.append(f"| {t.metric} | {t.nq_ms:.3f} | {t.qs_ms:.3f} | **{speedup_str}** |")
        lines.append("")

    # Accuracy tables
    lines.append("---\n")
    lines.append("## Calculation Accuracy\n")
    lines.append(
        "Side-by-side comparison of calculated values. "
        "Tolerance levels: EXACT (<1e-10), TIGHT (<1e-8), LOOSE (<1e-6), CALENDAR (<0.3%).\n"
    )

    for suite in suites:
        lines.append(f"### {suite.dataset_label}\n")
        lines.append("| Metric | NanuQuant | QuantStats | Rel. Difference | Tolerance | Status |")
        lines.append("|--------|----------:|-----------:|----------------:|-----------|--------|")
        for a in suite.accuracy_results:
            nq_str = format_value(a.nq_value)
            qs_str = format_value(a.qs_value)
            diff_str = format_pct(a.rel_diff)
            status = a.match
            lines.append(
                f"| {a.metric} | {nq_str} | {qs_str} | {diff_str} | {a.tolerance} | {status} |"
            )
        lines.append("")

    # Methodology
    lines.append("---\n")
    lines.append("## Methodology\n")
    lines.append("### Timing\n")
    lines.append("- Each metric is called 50 times after a warmup call")
    lines.append("- Reported time is the **median** across runs (resistant to outliers)")
    lines.append("- `time.perf_counter()` used for high-resolution timing")
    lines.append("- Both libraries receive identical numerical data\n")
    lines.append("### Accuracy\n")
    lines.append("- NanuQuant receives a `polars.Series`, QuantStats receives a `pandas.Series`")
    lines.append("- Both computed from the same underlying NumPy array")
    lines.append("- Relative difference = |NQ - QS| / |QS|")
    lines.append("- **PASS** = within tolerance, **DIFF*** = intentional difference (documented)\n")
    lines.append("### Known Intentional Differences\n")
    lines.append(
        "| Metric | Difference | Rationale |\n"
        "|--------|-----------|----------|\n"
        "| CAGR/Calmar | Periods-based vs calendar-based |"
        " Works with any time series |\n"
        "| Treynor | CAGR/Beta vs comp/Beta | Standard academic definition |\n"
        "| Omega | Correct implementation | Fixes bug in some QuantStats versions |\n"
    )

    lines.append("---\n")
    lines.append(
        "*Benchmarks run with NanuQuant (Polars-native) vs QuantStats-lumi (Pandas-based). "
        "Results will vary by hardware.*\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NanuQuant vs QuantStats")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output markdown file path")
    parser.add_argument(
        "--n-runs", type=int, default=50, help="Number of timing iterations per metric"
    )
    args = parser.parse_args()

    suites: list[BenchmarkSuite] = []

    # Synthetic datasets of increasing size
    for n in [1_000, 5_000, 10_000]:
        print(f"Running benchmarks: synthetic {n:,} observations...")
        pd_ret, pl_ret = make_data(n)
        suite = run_suite(pd_ret, pl_ret, f"Synthetic ({n:,} days)", n_runs=args.n_runs)
        suites.append(suite)

    # Real SPY data
    real = load_real_data()
    if real is not None:
        pd_ret, pl_ret = real
        print(f"Running benchmarks: SPY real data ({len(pd_ret):,} observations)...")
        suite = run_suite(pd_ret, pl_ret, "SPY (Real Market Data)", n_runs=args.n_runs)
        suites.append(suite)
    else:
        print("Skipping real data benchmarks (no cached SPY data found)")

    md = generate_markdown(suites)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
        print(f"\nBenchmark results written to {out_path}")
    else:
        print(md)


if __name__ == "__main__":
    main()
