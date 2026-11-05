"""Count discouraged phrases in the user-facing docs (report only).

The phrases are the "Writing documentation" list in CONTRIBUTING.md:
intensifiers, house jargon, narrative history and reader-as-audience voice.
Each hit is a prompt to reread the sentence, not an error; "exactly" before a
number or "lane" in a quoted identifier can be correct. Text inside fenced
code blocks and inline code spans is not counted.

Usage:
    python scripts/check_doc_style.py [--root REPO] [--verbose] [PATH ...]

With no paths given, scans site/src/**/*.md, README.md, crates/*/README.md
and crates/*/theory.md. Prints per-file counts (phrase hits and em-dashes),
then totals. --verbose also prints each hit with its line number.
Exit status is always 0: this is a report, not a gate.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

PHRASES = [
    # intensifiers and sincerity words
    "genuinely",
    "truly",
    "honestly",
    "honest",
    "actually",
    "really",
    "simply",
    "just",
    "deliberately",
    "crucially",
    "importantly",
    # house jargon
    "load-bearing",
    "lane",
    "lanes",
    "footgun",
    "the trap",
    "the catch",
    "smoking gun",
    "fingerprint",
    "sharp edge",
    "bites",
    # narrative history
    "turned out",
    "retracted",
    "the lesson",
    # reader-as-audience voice and signposting
    "note that",
    "worth noting",
    "in other words",
    "key insight",
    "let's",
    "here's",
    "this is the",
    "this means that",
    # marketing adjectives
    "high-performance",
    "blazing",
    "massive",
    "massively",
    "cutting-edge",
    "state-of-the-art",
    "seamless",
    "seamlessly",
    "world-class",
]

# Context exemptions: a hit is skipped when the pattern also matches here.
# "just" before a number or a comparison ("just above the cut-off") is
# precise, and "a note that ..." uses "note" as a noun.
EXEMPT = {
    "just": re.compile(
        r"\bjust\s+(?:[-+~]?\d|above|below|under|over|before|after)", re.IGNORECASE
    ),
    "note that": re.compile(r"\b(?:a|the|this)\s+note\s+that\b", re.IGNORECASE),
}
ASCII_DASH = re.compile(r"(?<=\s)--(?=\s)")

PATTERNS = {p: re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE) for p in PHRASES}
INLINE_CODE = re.compile(r"`[^`]*`")
FENCE = re.compile(r"^\s*(```|~~~)")


def prose_lines(text: str):
    """Yield (line number, line) outside fenced blocks, inline code removed."""
    in_fence = False
    for n, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield n, INLINE_CODE.sub("", line)


def scan(path: Path) -> tuple[Counter, int, list[str]]:
    hits: Counter = Counter()
    dashes = 0
    detail = []
    for n, line in prose_lines(path.read_text(encoding="utf-8")):
        dashes += line.count("—") + len(ASCII_DASH.findall(line))
        for phrase, pat in PATTERNS.items():
            k = len(pat.findall(line))
            if k and phrase in EXEMPT:
                k -= len(EXEMPT[phrase].findall(line))
            if k:
                hits[phrase] += k
                detail.append(f"    {n}: {phrase}: {line.strip()[:100]}")
    return hits, dashes, detail


def default_paths(root: Path) -> list[Path]:
    paths = sorted((root / "site/src").rglob("*.md"))
    paths.append(root / "README.md")
    paths += sorted(root.glob("crates/*/README.md"))
    paths += sorted(root.glob("crates/*/theory.md"))
    return [p for p in paths if p.is_file()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("paths", nargs="*", type=Path)
    args = ap.parse_args()
    paths = args.paths or default_paths(args.root)

    total: Counter = Counter()
    total_dashes = 0
    print(f"{'hits':>5} {'dash':>5}  file")
    for path in paths:
        hits, dashes, detail = scan(path)
        total += hits
        total_dashes += dashes
        try:
            name = path.resolve().relative_to(args.root.resolve())
        except ValueError:
            name = path
        top = ", ".join(f"{p} {c}" for p, c in hits.most_common(4))
        print(f"{sum(hits.values()):5d} {dashes:5d}  {name}  {top}")
        if args.verbose:
            print("\n".join(detail))
    print(f"{sum(total.values()):5d} {total_dashes:5d}  TOTAL ({len(paths)} files)")
    for phrase, c in total.most_common():
        print(f"    {c:5d}  {phrase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
