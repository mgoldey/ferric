#!/usr/bin/env python3
"""Stamp a release tag's version into the tree, for the macOS/Windows wheels.

    python scripts/wheels/stamp_version.py v0.1.0rc12

The macOS and Windows twin of wheels.yml build-wheels' "Stamp the version from
the tag" step (that step holds the rationale; keep the two in step). It is
Python rather than the Linux step's GNU sed because macOS ships BSD sed.
Standard library only; run from the repository root, on a clean checkout.

1. The tag must be canonical PEP 440 without hyphens or a local label:
   vX.Y.Z[(a|b|rc)N][.postN][.devN].
2. pyproject.toml's first ``version = `` line gets the PEP 440 version;
   crates/ferric-python/Cargo.toml's gets the SemVer twin (``-`` before the
   pre/post/dev parts, joined with ``.``: 0.1.0rc2 -> 0.1.0-rc2).
3. Verified with ``git diff``: exactly those two files, one line each,
   each equal to the expected line. Anything else exits 1.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

TAG_RE = re.compile(
    r"([0-9]+\.[0-9]+\.[0-9]+)((a|b|rc)[0-9]+)?(\.post[0-9]+)?(\.dev[0-9]+)?"
)
FILES = ("pyproject.toml", "crates/ferric-python/Cargo.toml")


def semver(version: str) -> str:
    """The SemVer twin of a canonical PEP 440 version (see module doc)."""
    m = TAG_RE.fullmatch(version)
    if not m:
        raise ValueError(version)
    extra = [p for p in (m.group(2), m.group(4), m.group(5)) if p]
    extra = [p.lstrip(".") for p in extra]
    return m.group(1) + ("-" + ".".join(extra) if extra else "")


def stamp_first_version_line(path: Path, version: str) -> None:
    # newline="": keep the file's own line endings on every OS.
    with path.open(encoding="utf-8", newline="") as fh:
        lines = fh.read().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("version = "):
            ending = line[len(line.rstrip("\r\n")) :]
            lines[i] = f'version = "{version}"{ending}'
            with path.open("w", encoding="utf-8", newline="") as fh:
                fh.write("".join(lines))
            return
    raise SystemExit(f"::error::no `version = ` line in {path}")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def main(argv: list[str]) -> int:
    if len(argv) != 1 or not argv[0].startswith("v"):
        print(__doc__, file=sys.stderr)
        return 2
    tag_version = argv[0][1:]
    if not TAG_RE.fullmatch(tag_version):
        print(
            f"::error::tag 'v{tag_version}' is not canonical PEP 440 without a "
            "local '+label' (no hyphens -- use v1.2.3rc1, not v1.2.3-rc1) -- "
            "refusing to stamp"
        )
        return 1
    semver_version = semver(tag_version)
    print(f"stamping {tag_version} (Cargo: {semver_version})")
    stamp_first_version_line(Path(FILES[0]), tag_version)
    stamp_first_version_line(Path(FILES[1]), semver_version)

    changed = sorted(git("diff", "--name-only").split())
    if changed != sorted(FILES):
        print(f"::error::stamping changed unexpected files: {changed}")
        return 1
    numstat = [ln.split("\t") for ln in git("diff", "--numstat").splitlines()]
    if any(n[:2] != ["1", "1"] for n in numstat):
        print(f"::error::stamp must change exactly one line per file: {numstat}")
        return 1
    diff = git("diff", "-U0").splitlines()
    added = sorted(ln for ln in diff if ln.startswith("+") and not ln.startswith("+++"))
    removed = [ln for ln in diff if ln.startswith("-") and not ln.startswith("---")]
    expected = sorted([f'+version = "{tag_version}"', f'+version = "{semver_version}"'])
    if added != expected:
        print(f"::error::stamped lines {added} differ from {expected}")
        return 1
    if any(not ln.startswith("-version = ") for ln in removed):
        print(f"::error::stamp removed a non-version line: {removed}")
        return 1
    print("stamp verified: only the two version lines changed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
