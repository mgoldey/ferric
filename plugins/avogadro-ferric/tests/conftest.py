"""Fixtures for the avogadro-ferric tests.

The shared helpers live in ``avoferric_testlib`` rather than here: test modules
import them by name, and ``from conftest import ...`` would resolve to
crates/ferric-python/tests/conftest.py when both suites run in one session.
"""

from avoferric_testlib import plugin_env  # noqa: F401  (pytest fixture)
