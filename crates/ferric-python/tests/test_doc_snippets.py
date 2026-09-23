"""The Python examples in the published docs run and print what the page says.

Each Markdown page with ``<!-- doctest -->`` blocks becomes one test case, run
by scripts/check_doc_snippets.py in a fresh interpreter and an empty working
directory. See that script's docstring for the marker syntax and the output
comparison rules.

The same check runs against the freshly built wheel in wheels.yml's smoke test;
this copy lets it run locally and in the nightly binding-test job.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

import ferric  # noqa: F401  (conftest turns a missing extension into a skip)

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts" / "check_doc_snippets.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("check_doc_snippets", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
PAGES = [
    p
    for p in runner.default_pages(ROOT)
    if runner.marked_blocks(p.read_text(encoding="utf-8"))
]


def test_some_pages_are_marked():
    # Guard against a vacuous suite: if the marker syntax or the page
    # discovery broke, every parametrized case below would silently vanish.
    assert len(PAGES) >= 2, f"expected several pages with doctest blocks, found {PAGES}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_doc_page_snippets(page):
    result = runner.run_page(page, ROOT)
    assert result.blocks > 0
    assert not result.failures, "\n\n".join(result.failures)
