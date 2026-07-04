"""Systemic risk and correlation metrics.

This module provides metrics for measuring systemic risk, tail dependence,
and regime-dependent correlations used in portfolio risk management.

References
----------
- Kritzman, M., Li, Y., Page, S., & Rigobon, R. (2010). "Principal Components
  as a Measure of Systemic Risk"
- Copula-based tail dependence measures
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import polars as pl

from nanuquant.core.utils import to_float_series
from nanuquant.core.validation import (
    validate_benchmark_match,
    validate_returns,
)


class AbsorptionRatioResult(NamedTuple):
    """Result of absorption ratio calculation.

    Attributes
    ----------
    absorption_ratio : float
        The absorption ratio (fraction of variance explained by top components).
    n_components : int
        Number of principal components used.
    total_assets : int
        Total number of assets in the analysis.
    eigenvalues : list[float]
        Eigenvalues of the correlation matrix.
    """

    absorption_ratio: float
    n_components: int
    total_assets: int
    eigenvalues: list[float]


def absorption_ratio(
    returns_matrix: pl.DataFrame,
    n_components: int | None = None,
    *,
    fraction: float = 0.2,
) -> AbsorptionRatioResult:
    """Calculate the Absorption Ratio (AR) for systemic risk.

    The Absorption Ratio measures systemic risk by calculating the fraction
    of total variance explained by a fixed number of top principal components.
    Higher AR indicates a more "tightly coupled" or fragile market.

    Parameters
    ----------
    returns_matrix : pl.DataFrame
        DataFrame where each column is an asset's return series.
        All columns must have the same length.
    n_components : int, optional
        Number of principal components to use. If None, uses
        fraction * n_assets (default 20%).
    fraction : float, default 0.2
        Fraction of assets to use for component count if n_components
        is not specified.

    Returns
    -------
    AbsorptionRatioResult
        Named tuple containing:
        - absorption_ratio: Fraction of variance explained (0 to 1)
        - n_components: Number of components used
        - total_assets: Number of assets
        - eigenvalues: All eigenvalues of correlation matrix

    Notes
    -----
    AR = sum(top k eigenvalues) / sum(all eigenvalues)

    Typical interpretation:
    - AR < 0.3: Low systemic risk, diversified market
    - AR 0.3-0.5: Moderate systemic risk
    - AR > 0.5: High systemic risk, fragile market

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> returns = pl.DataFrame({
    ...     "A": np.random.normal(0, 0.02, 100),
    ...     "B": np.random.normal(0, 0.02, 100),
    ...     "C": np.random.normal(0, 0.02, 100),
    ... })
    >>> result = absorption_ratio(returns)
    >>> 0 <= result.absorption_ratio <= 1
    True

    References
    ----------
    Kritzman, M., et al. (2010). "Principal Components as a Measure of
    Systemic Risk"
    """
    # Validate input
    if returns_matrix.is_empty():
        raise ValueError("Returns matrix is empty")

    n_assets = len(returns_matrix.columns)
    if n_assets < 2:
        raise ValueError("At least 2 assets required for absorption ratio")

    n_obs = len(returns_matrix)
    if n_obs < n_assets:
        raise ValueError(f"Need at least as many observations ({n_obs}) as assets ({n_assets})")

    # Determine number of components
    if n_components is None:
        n_components = max(1, int(fraction * n_assets))
    n_components = min(n_components, n_assets)

    # Convert to numpy for eigenvalue computation
    returns_np = returns_matrix.to_numpy()

    # Handle missing values
    if np.any(np.isnan(returns_np)):
        returns_np = np.nan_to_num(returns_np, nan=0.0)

    # Compute correlation matrix
    corr_matrix = np.corrcoef(returns_np, rowvar=False)

    # Handle edge cases where correlation is undefined
    if np.any(np.isnan(corr_matrix)):
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

    # Compute eigenvalues
    try:
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
    except np.linalg.LinAlgError:
        # Return neutral value if eigendecomposition fails
        return AbsorptionRatioResult(
            absorption_ratio=1.0 / n_assets,
            n_components=n_components,
            total_assets=n_assets,
            eigenvalues=[1.0] * n_assets,
        )

    # Ensure eigenvalues are non-negative (numerical stability)
    eigenvalues = np.maximum(eigenvalues, 0)

    total_variance = np.sum(eigenvalues)
    if total_variance == 0:
        return AbsorptionRatioResult(
            absorption_ratio=1.0 / n_assets,
            n_components=n_components,
            total_assets=n_assets,
            eigenvalues=eigenvalues.tolist(),
        )

    # Absorption ratio
    top_variance = np.sum(eigenvalues[:n_components])
    ar = top_variance / total_variance

    return AbsorptionRatioResult(
        absorption_ratio=float(ar),
        n_components=n_components,
        total_assets=n_assets,
        eigenvalues=eigenvalues.tolist(),
    )


def lower_tail_dependence(
    returns1: pl.Series,
    returns2: pl.Series,
    *,
    threshold_quantile: float = 0.05,
) -> float:
    """Calculate lower tail dependence between two return series.

    Lower tail dependence measures the probability that both assets
    experience extreme negative returns simultaneously (co-crash probability).

    Parameters
    ----------
    returns1 : pl.Series
        First return series.
    returns2 : pl.Series
        Second return series.
    threshold_quantile : float, default 0.05
        Quantile threshold for defining extreme events (left tail).

    Returns
    -------
    float
        Lower tail dependence coefficient (0 to 1).
        - 0: No tail dependence (independent in extreme events)
        - 1: Perfect tail dependence (always co-crash)

    Notes
    -----
    Empirical estimator:
        λ_L = P(r2 < VaR_q(r2) | r1 < VaR_q(r1))

    This is a non-parametric estimate based on observed frequencies.
    For more robust estimates with small samples, consider copula-based
    methods.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> r1 = pl.Series(np.random.normal(0, 0.02, 1000))
    >>> r2 = pl.Series(np.random.normal(0, 0.02, 1000))
    >>> ltd = lower_tail_dependence(r1, r2)
    >>> 0 <= ltd <= 1
    True
    """
    validate_returns(returns1)
    validate_returns(returns2)
    validate_benchmark_match(returns1, returns2)

    if not 0 < threshold_quantile < 0.5:
        raise ValueError("threshold_quantile must be between 0 and 0.5")

    returns1 = to_float_series(returns1)
    returns2 = to_float_series(returns2)

    # Calculate quantile thresholds
    q1 = returns1.quantile(threshold_quantile, interpolation="linear")
    q2 = returns2.quantile(threshold_quantile, interpolation="linear")

    if q1 is None or q2 is None:
        return 0.0

    # Count events where r1 is in left tail
    in_tail_1 = returns1 <= q1
    n_tail_1 = in_tail_1.sum()

    if n_tail_1 == 0:
        return 0.0

    # Count events where both are in left tail
    in_tail_both = in_tail_1 & (returns2 <= q2)
    n_tail_both = in_tail_both.sum()

    # Conditional probability
    tail_dependence = float(n_tail_both) / float(n_tail_1)

    return tail_dependence


def downside_correlation(
    returns1: pl.Series,
    returns2: pl.Series,
    *,
    threshold: float = 0.0,
) -> float:
    """Calculate correlation during downside regimes.

    Measures the correlation between two return series specifically
    during periods when the first series experiences negative returns.

    Parameters
    ----------
    returns1 : pl.Series
        First return series (used to define downside regime).
    returns2 : pl.Series
        Second return series.
    threshold : float, default 0.0
        Return threshold defining "downside" (default: negative returns).

    Returns
    -------
    float
        Pearson correlation during downside periods (-1 to 1).
        Returns NaN if insufficient downside observations.

    Notes
    -----
    Downside correlation often differs significantly from overall correlation.
    During market stress, correlations typically increase (correlation breakdown),
    which has important implications for portfolio risk management.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> r1 = pl.Series(np.random.normal(0, 0.02, 500))
    >>> r2 = pl.Series(np.random.normal(0, 0.02, 500))
    >>> dc = downside_correlation(r1, r2)
    >>> -1 <= dc <= 1 or np.isnan(dc)
    True
    """
    validate_returns(returns1)
    validate_returns(returns2)
    validate_benchmark_match(returns1, returns2)

    returns1 = to_float_series(returns1)
    returns2 = to_float_series(returns2)

    # Filter to downside periods
    downside_mask = returns1 < threshold
    r1_down = returns1.filter(downside_mask)
    r2_down = returns2.filter(downside_mask)

    if len(r1_down) < 2:
        return float("nan")

    # Calculate correlation using Polars' built-in function
    df = pl.DataFrame({"a": r1_down, "b": r2_down})
    correlation = df.select(pl.corr("a", "b")).item()

    if correlation is None:
        return float("nan")

    return float(correlation)


def upside_correlation(
    returns1: pl.Series,
    returns2: pl.Series,
    *,
    threshold: float = 0.0,
) -> float:
    """Calculate correlation during upside regimes.

    Measures the correlation between two return series specifically
    during periods when the first series experiences positive returns.

    Parameters
    ----------
    returns1 : pl.Series
        First return series (used to define upside regime).
    returns2 : pl.Series
        Second return series.
    threshold : float, default 0.0
        Return threshold defining "upside" (default: positive returns).

    Returns
    -------
    float
        Pearson correlation during upside periods (-1 to 1).
        Returns NaN if insufficient upside observations.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> r1 = pl.Series(np.random.normal(0, 0.02, 500))
    >>> r2 = pl.Series(np.random.normal(0, 0.02, 500))
    >>> uc = upside_correlation(r1, r2)
    >>> -1 <= uc <= 1 or np.isnan(uc)
    True
    """
    validate_returns(returns1)
    validate_returns(returns2)
    validate_benchmark_match(returns1, returns2)

    returns1 = to_float_series(returns1)
    returns2 = to_float_series(returns2)

    # Filter to upside periods
    upside_mask = returns1 > threshold
    r1_up = returns1.filter(upside_mask)
    r2_up = returns2.filter(upside_mask)

    if len(r1_up) < 2:
        return float("nan")

    # Calculate correlation using Polars' built-in function
    df = pl.DataFrame({"a": r1_up, "b": r2_up})
    correlation = df.select(pl.corr("a", "b")).item()

    if correlation is None:
        return float("nan")

    return float(correlation)


def _correlation_matrix_np(returns_matrix: pl.DataFrame) -> tuple[list[str], np.ndarray]:
    """Compute an all-pairs Pearson correlation matrix as a single matmul.

    The heavy O(n_obs * n_strategies^2) work is a dense matrix product, which
    is handed to BLAS via NumPy. Each column is demeaned and scaled to unit
    L2 norm so that ``Z.T @ Z`` is exactly the Pearson correlation matrix
    (the sample-size factor cancels between covariance and the two standard
    deviations, so no ``ddof`` term is required).

    Returns the column labels and the (n_strategies, n_strategies) correlation
    matrix. Strategies with zero variance yield NaN correlations against every
    other strategy (correlation is undefined for a constant series).
    """
    labels = returns_matrix.columns

    # (n_obs, n_strategies) float64 view. Polars hands NumPy a contiguous
    # buffer, so this is a cheap reshape rather than a per-cell copy.
    matrix = returns_matrix.to_numpy().astype(np.float64, copy=False)

    # Demean each column, then normalise it to unit length. After this,
    # the dot product of any two columns is their Pearson correlation.
    centered = matrix - matrix.mean(axis=0)
    norms = np.sqrt(np.einsum("ij,ij->j", centered, centered))

    # A constant strategy has an undefined correlation. Detect it by
    # peak-to-peak span rather than a zero norm: floating-point rounding in
    # the mean can leave a genuinely-flat series with a tiny non-zero norm.
    constant = matrix.max(axis=0) == matrix.min(axis=0)
    norms[constant] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        standardized = centered / norms

    corr = standardized.T @ standardized

    # Clamp tiny floating-point overshoots so results stay in [-1, 1].
    np.clip(corr, -1.0, 1.0, out=corr)
    return labels, corr


def correlation_matrix(returns_matrix: pl.DataFrame) -> pl.DataFrame:
    """Compute the full pairwise Pearson correlation matrix, vectorized.

    Designed for wide panels — thousands of strategies — where a per-pair
    loop would be prohibitively slow. The entire matrix is produced by a
    single BLAS matrix multiplication on standardized columns, so runtime
    scales with ``n_obs * n_strategies^2`` and stays sub-second even for
    thousands of strategies.

    Parameters
    ----------
    returns_matrix : pl.DataFrame
        Wide DataFrame where each column is one strategy's return series.
        All columns must share the same length and be pre-aligned in time.
        Inputs must be complete (no nulls); align/fill upstream if needed.

    Returns
    -------
    pl.DataFrame
        Square correlation matrix with a leading ``strategy`` label column.
        The value at row *i*, column *j* is the Pearson correlation between
        strategy *i* and strategy *j*. Zero-variance strategies produce NaN.

    See Also
    --------
    correlation_pairs : Long-form (tidy) unique pairs, with thresholding.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> df = pl.DataFrame(
    ...     {f"s{i}": np.random.normal(0, 0.02, 250) for i in range(500)}
    ... )
    >>> corr = correlation_matrix(df)
    >>> corr.shape
    (500, 501)
    """
    if returns_matrix.is_empty():
        raise ValueError("Returns matrix is empty")
    if len(returns_matrix.columns) < 2:
        raise ValueError("At least 2 strategies required for a correlation matrix")

    labels, corr = _correlation_matrix_np(returns_matrix)

    data: dict[str, object] = {"strategy": labels}
    data.update({label: corr[:, j] for j, label in enumerate(labels)})
    return pl.DataFrame(data)


def correlation_pairs(
    returns_matrix: pl.DataFrame,
    *,
    min_abs: float | None = None,
    sort: bool = True,
) -> pl.DataFrame:
    """Compute all unique strategy correlation pairs in long (tidy) form.

    Returns one row per unordered strategy pair (the upper triangle of the
    correlation matrix, excluding the self-correlation diagonal). The full
    matrix is computed with a single vectorized matmul; only the unique
    pairs that survive the optional ``min_abs`` filter are materialized into
    the result, which keeps memory bounded when most pairs are uninteresting.

    Parameters
    ----------
    returns_matrix : pl.DataFrame
        Wide DataFrame where each column is one strategy's return series.
        All columns must share the same length and be pre-aligned in time.
    min_abs : float, optional
        If given, keep only pairs whose absolute correlation is at least this
        value. Applied on the raw matrix *before* building the DataFrame, so a
        selective threshold avoids materializing the full N*(N-1)/2 pairs.
        NaN correlations (zero-variance strategies) are always dropped when a
        threshold is set.
    sort : bool, default True
        If True, sort the result by absolute correlation, descending, so the
        most correlated (and most anti-correlated) pairs appear first.

    Returns
    -------
    pl.DataFrame
        Columns ``strategy_a``, ``strategy_b``, ``correlation``. For N
        strategies there are N*(N-1)/2 rows before any threshold is applied
        (e.g. ~18M rows for 6000 strategies) — use ``min_abs`` to keep the
        output tractable.

    See Also
    --------
    correlation_matrix : Square matrix form.

    Notes
    -----
    For 6000 strategies the correlation matrix has ~18M unique pairs. The
    transient upper-triangle index arrays dominate memory (~0.7 GB); passing a
    ``min_abs`` threshold filters these down before the DataFrame is built.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> df = pl.DataFrame(
    ...     {f"s{i}": np.random.normal(0, 0.02, 250) for i in range(500)}
    ... )
    >>> pairs = correlation_pairs(df, min_abs=0.2)
    >>> set(pairs.columns) == {"strategy_a", "strategy_b", "correlation"}
    True
    """
    if returns_matrix.is_empty():
        raise ValueError("Returns matrix is empty")
    if len(returns_matrix.columns) < 2:
        raise ValueError("At least 2 strategies required for correlation pairs")

    labels, corr = _correlation_matrix_np(returns_matrix)
    labels_arr = np.asarray(labels)

    # Upper-triangle indices exclude the diagonal (self-correlations) and the
    # lower mirror, so each unordered pair appears exactly once.
    rows, cols = np.triu_indices(corr.shape[0], k=1)
    values = corr[rows, cols]

    if min_abs is not None:
        keep = np.abs(values) >= min_abs
        # Drop undefined correlations rather than let NaN slip past the filter.
        keep &= ~np.isnan(values)
        rows, cols, values = rows[keep], cols[keep], values[keep]

    pairs = pl.DataFrame(
        {
            "strategy_a": labels_arr[rows],
            "strategy_b": labels_arr[cols],
            "correlation": values,
        }
    )

    if sort:
        pairs = pairs.sort(pl.col("correlation").abs(), descending=True)

    return pairs
