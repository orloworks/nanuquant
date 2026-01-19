# Suggested Commands for NanuQuant Development

## Package Management (using uv)
```bash
# Install package in development mode
uv pip install -e ".[dev]"

# Install all optional dependencies
uv pip install -e ".[all]"

# Sync dependencies from lock file
uv sync
```

## Testing
```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_vs_quantstats.py -v

# Run differential tests against QuantStats
pytest tests/test_vs_quantstats.py -v

# Run integration tests with real market data
pytest -m integration

# Skip integration tests
pytest -m "not integration"

# Run benchmarks
pytest -m benchmark

# Skip benchmarks
pytest -m "not benchmark"

# Run tests with coverage
pytest --cov=nanuquant
```

## Type Checking
```bash
# Run mypy (strict mode configured in pyproject.toml)
mypy nanuquant
```

## Linting and Formatting
```bash
# Run ruff linter
ruff check nanuquant

# Run ruff with auto-fix
ruff check --fix nanuquant

# Format code with ruff
ruff format nanuquant
```

## Building
```bash
# Build package
uv build

# Build wheel only
uv build --wheel
```

## Git (Darwin/macOS)
```bash
# Standard git commands
git status
git add <files>
git commit -m "message"
git push
git pull
git log --oneline -10

# Branch management
git checkout -b <branch-name>
git checkout main
git merge <branch>
```

## File System (Darwin/macOS)
```bash
# List files
ls -la

# Find files
find . -name "*.py" -type f

# Search in files (use grep or ripgrep)
grep -r "pattern" nanuquant/
rg "pattern" nanuquant/
```
