"""`tools/` is a library; `experiments/` is a campaign. Keep them apart.

The rule, stated in `experiments/README.md`:

    tools/ must not import from experiments/, and must not hard-code a named
    molecule, target, or hypothesis.

Why it is worth a test rather than a convention: the danuglipron campaign's
analogue set originally lived in `tools/morph/design.py`, which made the library
un-reusable for any other molecule and made `tools/` tests fail whenever a
campaign hypothesis changed. Conventions decay silently; this fails loudly.

What is DELIBERATELY allowed: `tools/` docstrings cite campaign measurements
("measured 2026-08-29 on the danuglipron ensemble: ..."). A library rule
justified by a real observation beats an unattributed assertion, and a citation
is provenance, not a dependency. So this checks IMPORTS and EXECUTABLE CODE,
never prose.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

TOOLS = pathlib.Path(__file__).resolve().parents[1]
REPO = TOOLS.parent

# Names that would indicate a specific campaign leaked into the library.
CAMPAIGN_TOKENS = (
    "danuglipron", "DANUGLIPRON", "7LCJ", "GLP1R", "PF-06882961",
)


def _library_modules():
    for f in sorted(TOOLS.rglob("*.py")):
        if "__pycache__" in f.parts or "tests" in f.parts:
            continue
        yield f


def test_no_library_module_imports_from_experiments():
    """The hard rule. A library that imports a campaign cannot be reused."""
    offenders = []
    for f in _library_modules():
        tree = ast.parse(f.read_text())
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module)
            elif isinstance(n, ast.Import):
                mods += [a.name for a in n.names]
            for m in mods:
                if m.split(".")[0] == "experiments":
                    offenders.append(f"{f.relative_to(REPO)}:{n.lineno} imports {m}")
    assert not offenders, (
        "tools/ must not import from experiments/:\n  " + "\n  ".join(offenders)
    )


def test_no_campaign_names_in_library_executable_code():
    """Campaign names may appear in PROSE (provenance) but not in code.

    Docstrings and comments are excluded deliberately -- see the module
    docstring for why that exclusion is a feature, not a loophole.
    """
    offenders = []
    for f in _library_modules():
        src = f.read_text()
        tree = ast.parse(src)

        doc_lines: set[int] = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef)):
                if n.body and isinstance(n.body[0], ast.Expr) and \
                        isinstance(getattr(n.body[0], "value", None), ast.Constant) and \
                        isinstance(n.body[0].value.value, str):
                    s0 = n.body[0]
                    doc_lines.update(range(s0.lineno, (s0.end_lineno or s0.lineno) + 1))

        for i, line in enumerate(src.splitlines(), 1):
            if i in doc_lines or line.strip().startswith("#"):
                continue
            for tok in CAMPAIGN_TOKENS:
                if tok in line:
                    offenders.append(
                        f"{f.relative_to(REPO)}:{i}: {tok!r} in code -- "
                        f"{line.strip()[:60]}"
                    )
    assert not offenders, (
        "campaign-specific names in tools/ executable code:\n  "
        + "\n  ".join(offenders)
    )


def test_the_boundary_check_can_actually_fail():
    """Reachability: the detector must flag a real violation, or it is
    decorative. Builds the offending pattern in memory rather than writing a
    file into the tree."""
    src = "from experiments.danuglipron.design import danuglipron_analogues\n"
    tree = ast.parse(src)
    found = [
        n.module for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module
        and n.module.split(".")[0] == "experiments"
    ]
    assert found == ["experiments.danuglipron.design"]


def test_experiments_may_import_tools():
    """The allowed direction. If this ever fails, the split is upside down."""
    exp = REPO / "experiments"
    if not exp.is_dir():
        pytest.skip("no experiments/ directory")
    imports_tools = False
    for f in exp.rglob("*.py"):
        if "__pycache__" in f.parts:
            continue
        tree = ast.parse(f.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and \
                    n.module.split(".")[0] == "tools":
                imports_tools = True
    assert imports_tools, (
        "no experiment imports tools/ -- either the split is inverted or the "
        "campaigns are not using the shared machinery"
    )
