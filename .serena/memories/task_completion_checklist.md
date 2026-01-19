# Task Completion Checklist for NanuQuant

When completing a task, run through these steps:

## 1. Code Quality
```bash
# Type checking (must pass with no errors)
mypy nanuquant

# Linting (fix any issues)
ruff check nanuquant
ruff check --fix nanuquant

# Formatting
ruff format nanuquant
```

## 2. Testing
```bash
# Run all tests (must pass)
pytest

# If you modified institutional metrics, run specifically:
pytest tests/test_institutional_*.py -v

# If you modified core metrics, also run differential tests:
pytest tests/test_vs_quantstats.py -v
```

## 3. Documentation
- Update docstrings for any new/modified functions
- If adding new public API, update `nanuquant/__init__.py` exports
- If adding new metrics, update `__all__` list

## 4. Pre-commit Checklist
- [ ] mypy passes with no errors
- [ ] ruff check passes (or issues are intentional)
- [ ] pytest passes (all tests green)
- [ ] New functions have proper type hints
- [ ] New functions have NumPy-style docstrings
- [ ] Edge cases handled (empty input, insufficient data)
- [ ] If applicable: added to __all__ in module and package __init__.py
