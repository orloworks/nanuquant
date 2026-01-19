# Code Style and Conventions for NanuQuant

## General Style
- **Line length**: 100 characters max (configured in ruff)
- **Python version**: 3.10+ (use modern syntax)
- **Imports**: Use `from __future__ import annotations` for forward references

## Naming Conventions
- **Functions**: snake_case (e.g., `max_drawdown`, `volatility`)
- **Classes**: PascalCase (e.g., `MetricsConfig`, `Trade`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `VAR_SIGMA_MAP`, `DEFAULT_CONFIG`)
- **Private functions**: Prefix with underscore (e.g., `_validate_input`)

## Type Hints
- **Required**: All public functions must have full type hints
- **Style**: Use modern syntax (`list[int]` not `List[int]`)
- **Optional params**: Use `param: int | None = None` syntax
- **Return types**: Always specify, use `-> None` for void functions

## Docstrings
- **Format**: NumPy-style docstrings
- **Required sections**: 
  - Description (brief one-liner)
  - Parameters (with types)
  - Returns (with type)
- **Optional sections**: Notes, Examples, Raises

Example:
```python
def volatility(
    returns: pl.Series,
    *,
    periods_per_year: int | None = None,
    annualize: bool = True,
) -> float:
    """Calculate return volatility (standard deviation).

    Parameters
    ----------
    returns : pl.Series
        Period returns.
    periods_per_year : int, optional
        Number of periods per year for annualization. If None, uses config default.
    annualize : bool, default True
        If True, annualize the volatility using sqrt(periods_per_year).

    Returns
    -------
    float
        Annualized (or raw) volatility of returns.

    Notes
    -----
    Formula: std(returns) * sqrt(periods_per_year)

    Examples
    --------
    >>> import polars as pl
    >>> returns = pl.Series([0.01, -0.02, 0.015, -0.01, 0.02])
    >>> volatility(returns, periods_per_year=252)
    0.248...
    """
```

## Function Signatures
- Use keyword-only arguments after `*` for clarity
- Default `periods_per_year` to `None` and use config
- Return `float` for scalar metrics, `pl.Series` for arrays

## Input Validation
- Use `validate_returns()` at function start
- Use `validate_min_length()` for minimum data requirements
- Use `to_float_series()` to ensure float dtype

## Error Handling
- Use custom exceptions from `nanuquant.exceptions`
- `EmptySeriesError` for empty input
- `InsufficientDataError` for not enough data points
- Return `0.0` or `float('nan')` for edge cases (match QuantStats behavior)

## Testing Patterns
- Use pytest fixtures from `conftest.py`
- Differential tests against QuantStats for validation
- Test edge cases: empty, single value, all positive, all negative
- Use `pytest.mark.integration` for real market data tests
- Use `pytest.mark.benchmark` for performance tests

## Configuration
- Use `get_config()` to access global settings
- Default periods_per_year: 252 (daily trading days)
- Default ddof: 1 (sample standard deviation)
