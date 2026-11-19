"""Layering boundary: ferric's Python package imports nothing above it.

ferric is the bottom of the stack for its Python users. Applications built on
it (``smeltery``) and provenance tooling (``attestation``) depend on ferric;
ferric must never depend on them, or the dependency cycle makes ferric
unusable on its own. An AST scan of every module under
``crates/ferric-python/python/ferric/`` rejects an import of either, in any
spelling (``import x``, ``import x.y``, ``from x import y``,
``from x.y import z``, lazy imports inside functions included).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "python" / "ferric"
FORBIDDEN = frozenset({"smeltery", "attestation"})


def forbidden_imports(source: str, filename: str = "<string>") -> list[tuple[int, str]]:
    """``(line, module)`` for each import of a forbidden top-level package."""
    found = []
    for node in ast.walk(ast.parse(source, filename)):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        found += [(node.lineno, n) for n in names if n.split(".")[0] in FORBIDDEN]
    return sorted(found)


def test_scan_covers_the_package():
    files = sorted(PACKAGE.rglob("*.py"))
    assert len(files) >= 5 and PACKAGE / "__init__.py" in files


def test_ferric_python_imports_nothing_above_it():
    offenders = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        hits = forbidden_imports(path.read_text(encoding="utf-8"), str(path))
        if hits:
            offenders[str(path.relative_to(PACKAGE))] = hits
    assert not offenders, f"ferric must not import smeltery or attestation: {offenders}"


@pytest.mark.parametrize(
    "source",
    [
        "import smeltery",
        "import attestation.mcp",
        "import os, smeltery.core as s",
        "from smeltery import run",
        "from attestation.ledger import scan",
        "def f():\n    import smeltery\n",
        "try:\n    from smeltery.x import y\nexcept ImportError:\n    pass\n",
    ],
)
def test_detector_can_fail(source):
    assert forbidden_imports(source), source


@pytest.mark.parametrize(
    "source",
    [
        "import os\nimport numpy as np",
        "from . import smeltery_like",
        "from .smeltery import x",  # a relative import is ferric's own module
        "import smeltery_fake",
        "# import smeltery\nx = 'import smeltery'",
    ],
)
def test_detector_does_not_overreach(source):
    assert forbidden_imports(source) == []
