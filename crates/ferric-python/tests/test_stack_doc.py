"""docs/stack.md is the stack-overview page mirrored in ferric, smeltery and attestation.

CI sees one checkout, so it cannot compare the three copies. It pins the version line
instead: the page says it must be edited in all three repositories together, and a change
made here alone fails this test until the constant (and the other repos' constants) move.

The book (mdBook, site/) shows the page through an include, so there is one copy in this
repository. The page must also be reachable from the README and the book's SUMMARY.md.
This module never touches the extension, so conftest.py lists it in EXTENSION_FREE.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STACK_HEADER = "<!-- stack-overview v1 2026-11-20 -->"


def test_the_stack_overview_carries_the_pinned_version_line():
    page = ROOT / "docs" / "stack.md"
    assert page.is_file(), "docs/stack.md is missing"
    assert page.read_text(encoding="utf-8").splitlines()[0] == STACK_HEADER, (
        "docs/stack.md is mirrored in ferric, smeltery and attestation: update all three "
        "and bump STACK_HEADER in each repository's test"
    )


def test_the_book_and_readme_reach_the_stack_overview():
    assert "](docs/stack.md)" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "(./stack.md)" in (ROOT / "site" / "src" / "SUMMARY.md").read_text(
        encoding="utf-8"
    )
    include = (ROOT / "site" / "src" / "stack.md").read_text(encoding="utf-8")
    assert "{{#include ../../docs/stack.md}}" in include
