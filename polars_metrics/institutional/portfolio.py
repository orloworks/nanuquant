"""Portfolio risk decomposition metrics.

This module provides metrics for decomposing portfolio risk into component
contributions, including marginal contribution to risk and covariance
shrinkage estimators.

References
----------
- Euler decomposition for risk attribution
- Ledoit, O., & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix"
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import polars as pl



class MCRResult(NamedTuple):
    """Result of marginal contribution to risk calculation.

    Attributes
    ----------
    mcr : list[float]
        Marginal contribution to risk for each asset.
    pcr : list[float]
        Percentage contribution to risk for each asset (sums to 1).
    total_risk : float
        Total portfolio volatility.
    asset_names : list[str]
        Names of assets (column names from input).
    """

    mcr: list[float]
    pcr: list[float]
    total_risk: float
    asset_names: list[str]


class LedoitWolfResult(NamedTuple):
    """Result of Ledoit-Wolf covariance shrinkage.

    Attributes
    ----------
    covariance : np.ndarray
        Shrunk covariance matrix.
    shrinkage_intensity : float
        Optimal shrinkage intensity (0 = sample, 1 = target).
    sample_covariance : np.ndarray
        Original sample covariance matrix.
    """

    covariance: np.ndarray
    shrinkage_intensity: float
    sample_covariance: np.ndarray


def marginal_contribution_to_risk(
    returns_matrix: pl.DataFrame,
    weights: list[float] | np.ndarray,
) -> MCRResult:
    """Calculate marginal contribution to risk using Euler decomposition.

    Decomposes total portfolio risk into the contribution from each asset,
    which is essential for risk budgeting and portfolio construction.

    Parameters
    ----------
    returns_matrix : pl.DataFrame
        DataFrame where each column is an asset's return series.
    weights : list[float] or np.ndarray
        Portfolio weights for each asset. Must sum to 1 (or any constant).

    Returns
    -------
    MCRResult
        Named tuple containing:
        - mcr: Marginal contribution to risk for each asset
        - pcr: Percentage contribution to risk (sums to 1)
        - total_risk: Total portfolio volatility
        - asset_names: Asset names from column headers

    Notes
    -----
    The marginal contribution to risk (MCR) for asset i is:

        MCR_i = (Σw)_i / σ_p

    Where Σ is the covariance matrix, w is the weight vector, and σ_p
    is the portfolio volatility.

    The percentage contribution to risk (PCR) is:

        PCR_i = w_i * MCR_i / σ_p

    The sum of PCR equals 1 (full attribution).

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> returns = pl.DataFrame({
    ...     "A": np.random.normal(0, 0.02, 100),
    ...     "B": np.random.normal(0, 0.015, 100),
    ... })
    >>> result = marginal_contribution_to_risk(returns, [0.6, 0.4])
    >>> abs(sum(result.pcr) - 1.0) < 0.01
    True
    """
    # Validate input
    if returns_matrix.is_empty():
        raise ValueError("Returns matrix is empty")

    weights = np.array(weights, dtype=float)
    n_assets = len(returns_matrix.columns)

    if len(weights) != n_assets:
        raise ValueError(
            f"Weight count ({len(weights)}) must match asset count ({n_assets})"
        )

    asset_names = returns_matrix.columns

    # Convert to numpy
    returns_np = returns_matrix.to_numpy()

    # Handle missing values
    if np.any(np.isnan(returns_np)):
        returns_np = np.nan_to_num(returns_np, nan=0.0)

    # Calculate covariance matrix
    cov_matrix = np.cov(returns_np, rowvar=False)

    # Ensure 2D for single asset edge case
    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[cov_matrix]])

    # Portfolio variance and volatility
    port_variance = weights @ cov_matrix @ weights
    if port_variance <= 0:
        return MCRResult(
            mcr=[0.0] * n_assets,
            pcr=[1.0 / n_assets] * n_assets,
            total_risk=0.0,
            asset_names=list(asset_names),
        )

    port_vol = math.sqrt(port_variance)

    # Marginal contribution to risk
    mcr = (cov_matrix @ weights) / port_vol

    # Percentage contribution to risk
    pcr = weights * mcr / port_vol

    # Normalize PCR to sum to 1 (numerical stability)
    pcr_sum = np.sum(pcr)
    if pcr_sum > 0:
        pcr = pcr / pcr_sum

    return MCRResult(
        mcr=mcr.tolist(),
        pcr=pcr.tolist(),
        total_risk=float(port_vol),
        asset_names=list(asset_names),
    )


def ledoit_wolf_covariance(
    returns_matrix: pl.DataFrame,
) -> LedoitWolfResult:
    """Calculate Ledoit-Wolf shrinkage covariance estimator.

    The Ledoit-Wolf estimator shrinks the sample covariance matrix toward
    a structured target (scaled identity matrix) to reduce estimation error,
    especially beneficial for high-dimensional or noisy data.

    Parameters
    ----------
    returns_matrix : pl.DataFrame
        DataFrame where each column is an asset's return series.

    Returns
    -------
    LedoitWolfResult
        Named tuple containing:
        - covariance: Shrunk covariance matrix
        - shrinkage_intensity: Optimal shrinkage parameter (0-1)
        - sample_covariance: Original sample covariance

    Notes
    -----
    The shrunk estimator is:
        Σ_shrunk = δ * F + (1 - δ) * S

    Where:
    - S is the sample covariance matrix
    - F is the shrinkage target (scaled identity)
    - δ is the optimal shrinkage intensity

    The shrinkage intensity is determined analytically to minimize
    expected quadratic loss.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> returns = pl.DataFrame({
    ...     "A": np.random.normal(0, 0.02, 100),
    ...     "B": np.random.normal(0, 0.015, 100),
    ...     "C": np.random.normal(0, 0.025, 100),
    ... })
    >>> result = ledoit_wolf_covariance(returns)
    >>> 0 <= result.shrinkage_intensity <= 1
    True

    References
    ----------
    Ledoit, O., & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix"
    """
    # Validate input
    if returns_matrix.is_empty():
        raise ValueError("Returns matrix is empty")

    # Convert to numpy
    X = returns_matrix.to_numpy()
    n, p = X.shape

    if n < 2:
        raise ValueError("Need at least 2 observations")

    # Handle missing values
    if np.any(np.isnan(X)):
        X = np.nan_to_num(X, nan=0.0)

    # Demean
    X = X - X.mean(axis=0)

    # Sample covariance
    sample_cov = (X.T @ X) / n

    # Shrinkage target: scaled identity matrix
    # Use average variance as the scaling factor
    mu = np.trace(sample_cov) / p
    F = mu * np.eye(p)

    # Compute shrinkage intensity using Ledoit-Wolf formula
    delta = _compute_shrinkage_intensity(X, sample_cov, F, n, p)

    # Shrunk covariance
    shrunk_cov = delta * F + (1 - delta) * sample_cov

    return LedoitWolfResult(
        covariance=shrunk_cov,
        shrinkage_intensity=float(delta),
        sample_covariance=sample_cov,
    )


def _compute_shrinkage_intensity(
    X: np.ndarray,
    sample_cov: np.ndarray,
    F: np.ndarray,
    n: int,
    p: int,
) -> float:
    """Compute optimal Ledoit-Wolf shrinkage intensity.

    Uses the analytical formula from Ledoit & Wolf (2004).
    """
    # Squared Frobenius norm of sample covariance minus target
    delta_sq = np.sum((sample_cov - F) ** 2)

    if delta_sq == 0:
        return 0.0

    # Compute sum of squared sample covariances
    # This is used in the numerator of the shrinkage formula

    # Average squared entry of sample covariance
    mu = np.trace(sample_cov) / p

    # Vectorized calculation of the sum of squared differences
    # This replaces: sum_{i=1 to n} || x_i x_i^T - S ||_F^2
    # where x_i is a row vector of demeaned returns
    term1 = np.sum(np.sum(X**2, axis=1) ** 2)
    term2 = -2 * np.sum(np.diag(X @ sample_cov @ X.T))
    term3 = n * np.sum(sample_cov**2)
    sum_sq = term1 + term2 + term3

    # Asymptotic estimate of optimal shrinkage
    # Division by n^2: first /n for mean, second /n for proper scaling
    kappa = (sum_sq / n**2) / delta_sq

    # Bound between 0 and 1
    delta = min(1.0, max(0.0, kappa))

    return delta


def correlation_from_covariance(
    covariance: np.ndarray,
) -> np.ndarray:
    """Convert a covariance matrix to correlation matrix.

    Parameters
    ----------
    covariance : np.ndarray
        Covariance matrix.

    Returns
    -------
    np.ndarray
        Correlation matrix.
    """
    std = np.sqrt(np.diag(covariance))
    outer_std = np.outer(std, std)

    # Avoid division by zero
    outer_std = np.where(outer_std == 0, 1, outer_std)

    correlation = covariance / outer_std

    # Ensure diagonal is exactly 1
    np.fill_diagonal(correlation, 1.0)

    return correlation


def portfolio_volatility(
    returns_matrix: pl.DataFrame,
    weights: list[float] | np.ndarray,
    *,
    use_shrinkage: bool = False,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> float:
    """Calculate portfolio volatility.

    Parameters
    ----------
    returns_matrix : pl.DataFrame
        DataFrame where each column is an asset's return series.
    weights : list[float] or np.ndarray
        Portfolio weights for each asset.
    use_shrinkage : bool, default False
        If True, use Ledoit-Wolf shrinkage estimator for covariance.
    annualize : bool, default True
        If True, annualize the volatility.
    periods_per_year : int, default 252
        Periods per year for annualization.

    Returns
    -------
    float
        Portfolio volatility (annualized if annualize=True).

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> returns = pl.DataFrame({
    ...     "A": np.random.normal(0, 0.02, 100),
    ...     "B": np.random.normal(0, 0.015, 100),
    ... })
    >>> vol = portfolio_volatility(returns, [0.6, 0.4])
    >>> vol > 0
    True
    """
    if returns_matrix.is_empty():
        return 0.0

    weights = np.array(weights, dtype=float)
    returns_np = returns_matrix.to_numpy()

    if np.any(np.isnan(returns_np)):
        returns_np = np.nan_to_num(returns_np, nan=0.0)

    # Get covariance matrix
    if use_shrinkage:
        result = ledoit_wolf_covariance(returns_matrix)
        cov_matrix = result.covariance
    else:
        cov_matrix = np.cov(returns_np, rowvar=False)

    # Ensure 2D
    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[cov_matrix]])

    # Portfolio variance
    port_var = weights @ cov_matrix @ weights

    if port_var <= 0:
        return 0.0

    vol = math.sqrt(port_var)

    if annualize:
        vol *= math.sqrt(periods_per_year)

    return float(vol)
