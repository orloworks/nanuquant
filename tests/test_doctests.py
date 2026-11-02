"""The docstring examples are documentation of the library's accuracy: run them.

Covers the modules whose examples have been checked; add a module here once its
examples pass.
"""

from __future__ import annotations

import doctest

import pytest

from nanuquant.core import performance, risk

FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE


@pytest.mark.parametrize("module", [risk, performance], ids=lambda m: m.__name__)
def test_docstring_examples(module) -> None:
    result = doctest.testmod(module, optionflags=FLAGS, verbose=False)
    assert result.attempted > 0
    assert result.failed == 0, f"{result.failed} docstring example(s) failed in {module.__name__}"
