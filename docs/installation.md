# Installation Guide

> **Disclaimer**: NanuQuant is for educational and research purposes only. See [DISCLAIMER.md](../DISCLAIMER.md) for important legal notices. Nothing in this library constitutes financial advice.

## Requirements

- **Python**: 3.10 or higher
- **Operating Systems**: Linux, macOS, Windows

## Quick Install

Install NanuQuant from PyPI:

```bash
pip install nanuquant
```

## Installation Options

### Standard Installation

For basic usage with core functionality:

```bash
pip install nanuquant
```

This installs:
- `polars>=0.20.0` - High-performance DataFrame library
- `scipy>=1.10.0` - Scientific computing (for statistical tests)
- `numpy>=1.24` - Numerical computing

### With Reports Support

For HTML report generation:

```bash
pip install nanuquant[reports]
```

Additional dependency:
- `msgspec>=0.18` - Fast serialization for report generation

### Development Installation

For contributors or those wanting to run tests:

```bash
pip install nanuquant[dev]
```

Additional dependencies:
- `pytest>=8.0` - Testing framework
- `pytest-benchmark>=4.0` - Performance benchmarking
- `quantstats-lumi>=0.3` - For differential testing
- `pandas>=2.0` - Required for QuantStats comparison tests
- `mypy>=1.8` - Static type checking
- `yfinance>=0.2.40` - Market data for integration tests

### Full Installation

Install all optional dependencies:

```bash
pip install nanuquant[all]
```

## Installing from Source

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/launchstack-dev/nanuquant.git
cd nanuquant

# Install in development mode
pip install -e .

# Or with all dependencies
pip install -e ".[all]"
```

### Using a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install NanuQuant
pip install nanuquant
```

### Using Poetry

```bash
# Add to your project
poetry add nanuquant

# With optional dependencies
poetry add nanuquant[reports]
```

### Using Conda

```bash
# Create environment
conda create -n quant python=3.11
conda activate quant

# Install via pip (NanuQuant is not on conda-forge yet)
pip install nanuquant
```

## Verifying Installation

After installation, verify NanuQuant is working:

```python
import nanuquant as nq
import polars as pl

# Check version
print(f"NanuQuant version: {nq.__version__}")

# Quick test
returns = pl.Series("returns", [0.01, -0.02, 0.03, 0.01, -0.01])
sharpe = nq.sharpe(returns)
print(f"Sharpe ratio: {sharpe:.4f}")
```

Expected output:
```
NanuQuant version: 0.1.0
Sharpe ratio: 0.4472
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'polars'

```bash
pip install polars>=0.20.0
```

#### Version Conflicts

If you have version conflicts with other packages:

```bash
# Create a fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows
pip install nanuquant
```

#### Apple Silicon (M1/M2) Macs

NanuQuant works natively on Apple Silicon. If you encounter issues:

```bash
# Ensure you're using an ARM-native Python
python -c "import platform; print(platform.machine())"
# Should output: arm64

# Install with native wheels
pip install nanuquant --no-cache-dir
```

#### Windows Long Path Issues

If you encounter path length issues on Windows:

1. Enable long paths in Windows (requires admin):
   ```powershell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```

2. Or use a shorter installation path

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/launchstack-dev/nanuquant/issues)
2. Ensure you're using Python 3.10+
3. Try installing in a fresh virtual environment
4. Verify all dependencies are compatible

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade nanuquant
```

## Uninstalling

```bash
pip uninstall nanuquant
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Get started with basic usage
- [API Reference](api/core.md) - Detailed function documentation
- [Use Cases](use-cases.md) - Practical examples and applications
