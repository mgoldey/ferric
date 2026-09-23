"""Check every internal link and #anchor in a built mdBook site.

mdBook's own build only fails on a missing SUMMARY.md entry. A link from one
page to another page that does not exist, or to a heading that was renamed,
builds cleanly and 404s for the reader. This walks the rendered HTML (so it
sees the anchors mdBook actually generated, including for headings with
Unicode or code spans) and checks each relative href in the page body.

Usage:
    mdbook build site -d /tmp/book
    python scripts/check_doc_links.py /tmp/book

Links into api/ are skipped: rustdoc output is added to the site by the Pages
workflow, not by `mdbook build`. External (scheme://) links are not checked.
Exit status is 0 when every link resolves, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

HREF = re.compile(r'href="([^"]+)"')
ID = re.compile(r'id="([^"]+)"')
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def page_body(html: str) -> str:
    """The <main> element, so the sidebar and theme chrome are not checked."""
    start, end = html.find("<main>"), html.find("</main>")
    return html[start:end] if start != -1 and end != -1 else html


def check(book: Path) -> list[str]:
    pages = sorted(book.rglob("*.html"))
    ids = {p.resolve(): set(ID.findall(p.read_text(encoding="utf-8"))) for p in pages}
    problems: list[str] = []
    for page in pages:
        # print.html concatenates every page and repeats each link; checking
        # the individual pages is enough and keeps the report readable.
        if page.name == "print.html" or "api" in page.relative_to(book).parts:
            continue
        for href in HREF.findall(page_body(page.read_text(encoding="utf-8"))):
            if SCHEME.match(href):
                continue
            path, _, frag = href.partition("#")
            target = page if not path else (page.parent / unquote(path))
            rel_target = target.resolve()
            try:
                if "api" in rel_target.relative_to(book.resolve()).parts:
                    continue
            except ValueError:
                problems.append(
                    f"{page.relative_to(book)}: {href} points outside the book"
                )
                continue
            if path and not rel_target.exists():
                problems.append(f"{page.relative_to(book)}: {href} (no such file)")
                continue
            if (
                frag
                and rel_target.suffix == ".html"
                and unquote(frag) not in ids.get(rel_target, set())
            ):
                problems.append(f"{page.relative_to(book)}: {href} (no such anchor)")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    book = Path(argv[1])
    if not (book / "index.html").is_file():
        print(f"{book} does not look like a built mdBook (no index.html)")
        return 2
    problems = check(book)
    n_pages = sum(1 for _ in book.rglob("*.html"))
    for p in problems:
        print(p)
    print(f"{n_pages} HTML files checked, {len(problems)} broken link(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
