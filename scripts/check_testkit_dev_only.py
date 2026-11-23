#!/usr/bin/env python3
"""Fail if any crate takes `ferric-testkit` as anything but a dev-dependency.

The testkit is test-support code (`publish = false`). A normal or build
dependency on it would put test helpers in a shipped artifact and, because it
depends on ferric-core, can introduce a dependency cycle. The only manifest
allowed to name it outside [dev-dependencies] is its own.

Usage: python3 scripts/check_testkit_dev_only.py [--root DIR]
Exit status: 0 pass, 1 fail.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

NAME = "ferric-testkit"


def offending_lines(manifest: Path):
    """Yield (line number, section) for each mention outside dev-dependencies."""
    section = ""
    for n, raw in enumerate(manifest.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        m = re.match(r"\[(.+)\]$", line)
        if m:
            section = m.group(1).strip()
            continue
        if re.match(rf'"?{re.escape(NAME)}"?\s*(=|\.)', line):
            if "dev-dependencies" not in section:
                yield n, section
        # `[dependencies.ferric-testkit]` table form.
        if (
            section.endswith(f"dependencies.{NAME}")
            and "dev-dependencies" not in section
        ):
            yield n, section


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    root = ap.parse_args(argv).root
    bad = []
    manifests = [
        root / "Cargo.toml",
        *sorted(root.glob("crates/*/Cargo.toml")),
        *sorted(root.glob("benchmarks/*/Cargo.toml")),
    ]
    for mf in manifests:
        if not mf.is_file() or mf.parent.name == NAME:
            continue
        for n, section in offending_lines(mf):
            bad.append(
                f"{mf.relative_to(root)}:{n}: {NAME} in [{section}]; use [dev-dependencies]"
            )
    for b in sorted(set(bad)):
        print("FAIL:", b)
    if not bad:
        print(f"{NAME} is a dev-dependency everywhere: OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
