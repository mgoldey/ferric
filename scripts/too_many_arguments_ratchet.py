#!/usr/bin/env python3
"""Ratchet on `clippy::too_many_arguments` allows: the count can only go down.

Two halves, both needed:

  * The compiler half. `[workspace.lints.clippy] too_many_arguments` must not be
    "allow" (root Cargo.toml), and no member manifest may override it, so
    `cargo clippy -- -D warnings` rejects any NEW function over clippy's
    7-argument limit unless it carries an item-level
    `#[allow(clippy::too_many_arguments)]`.
  * The counting half (this script). It counts those item-level allows per
    crate and compares them with scripts/too_many_arguments_baseline.json. A
    crate over its baseline fails: the fix is to bundle the arguments into a
    struct, not to raise the baseline. A crate UNDER its baseline only prints
    a notice with the `--update-baseline` command (exit 0, so two PRs that
    each lower the same crate cannot turn main red after a clean merge);
    `--strict` makes it fatal, for local use.

Counted: `#[allow(...)]`, `#[expect(...)]` and the same inside
`#[cfg_attr(..., allow(...))]`, with the lint anywhere in the list. Always an
error: a crate-level (`#![...]`) allow of the lint (it would hide every new
offender), and an item- or crate-level allow of the `clippy::complexity` or
`clippy::all` groups (they contain the lint).

Usage:
    python3 scripts/too_many_arguments_ratchet.py                  # check
    python3 scripts/too_many_arguments_ratchet.py --strict         # also fail if under
    python3 scripts/too_many_arguments_ratchet.py --update-baseline  # lower only
    python3 scripts/too_many_arguments_ratchet.py --root DIR       # other checkout

Exit status: 0 pass, 1 fail.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

LINT = "clippy::too_many_arguments"
BASELINE_REL = "scripts/too_many_arguments_baseline.json"

# Any attribute, inner or outer, possibly spanning lines. The lint lists are
# then pulled out of every `allow(...)` / `expect(...)` in it, which covers
# `#[cfg_attr(feature = "x", allow(...))]` too.
ATTR = re.compile(r"#(?P<bang>!?)\[(?P<body>[^\]]*)\]", re.S)
LINT_LIST = re.compile(r"\b(?:allow|expect)\(([^()]*)\)")
GROUPS = ("clippy::complexity", "clippy::all")
LINE_COMMENT = re.compile(r"//[^\n]*")


def crate_of(root: Path, path: Path) -> str:
    rel = path.relative_to(root).parts
    if rel[0] == "crates":
        return rel[1]
    return rel[0]


def rust_files(root: Path):
    for top in ("crates", "benchmarks"):
        base = root / top
        if base.is_dir():
            for p in sorted(base.rglob("*.rs")):
                # Never descend into build output kept inside the tree.
                if "target" in p.relative_to(root).parts:
                    continue
                yield p


def count(root: Path):
    """Return ({crate: item-level allows}, [crate-level allow locations],
    [group-allow locations])."""
    counts: dict[str, int] = {}
    crate_level: list[str] = []
    groups: list[str] = []
    for path in rust_files(root):
        text = LINE_COMMENT.sub("", path.read_text(encoding="utf-8", errors="replace"))
        for m in ATTR.finditer(text):
            lints = [
                x.strip()
                for lm in LINT_LIST.finditer(m.group("body"))
                for x in lm.group(1).split(",")
            ]
            loc = f"{path.relative_to(root)}:{text.count(chr(10), 0, m.start()) + 1}"
            for g in GROUPS:
                if g in lints:
                    groups.append(f"{loc} ({g})")
            if LINT not in lints:
                continue
            crate = crate_of(root, path)
            if m.group("bang"):
                crate_level.append(loc)
            else:
                counts[crate] = counts.get(crate, 0) + 1
    return counts, crate_level, groups


def lint_config_problems(root: Path):
    """Manifest settings that would switch the compiler half off."""
    problems = []
    manifests = [root / "Cargo.toml", *sorted((root / "crates").glob("*/Cargo.toml"))]
    for mf in manifests:
        if not mf.is_file():
            continue
        for n, line in enumerate(mf.read_text().splitlines(), 1):
            s = line.split("#", 1)[0].strip()
            if re.match(
                r"too_many_arguments\s*=\s*(\"allow\"|\{[^}]*level\s*=\s*\"allow\")", s
            ):
                problems.append(f"{mf.relative_to(root)}:{n}: {s}")
    return problems


def load_baseline(root: Path) -> dict[str, int]:
    p = root / BASELINE_REL
    if not p.is_file():
        return {}
    return {k: int(v) for k, v in json.loads(p.read_text())["crates"].items()}


def write_baseline(root: Path, counts: dict[str, int]):
    data = {
        "_comment": (
            "Item-level #[allow(clippy::too_many_arguments)] per crate. Only ever "
            "lowered: scripts/too_many_arguments_ratchet.py --update-baseline."
        ),
        "crates": dict(sorted(counts.items())),
    }
    (root / BASELINE_REL).write_text(json.dumps(data, indent=2) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="also fail when a crate is under its baseline",
    )
    args = ap.parse_args(argv)
    root = args.root

    counts, crate_level, groups = count(root)
    base = load_baseline(root)
    errors = []

    for loc in crate_level:
        errors.append(
            f"crate-level allow of {LINT} at {loc}: convert it to item-level allows"
        )
    for loc in groups:
        errors.append(
            f"allow of a clippy group that contains {LINT} at {loc}: allow the specific lints instead"
        )
    for p in lint_config_problems(root):
        errors.append(f"{LINT} is allowed in a manifest: {p}")

    if args.update_baseline:
        raised = {
            c: (base.get(c, 0), n) for c, n in counts.items() if n > base.get(c, 0)
        }
        if base or (root / BASELINE_REL).exists():
            for c, (old, new) in sorted(raised.items()):
                errors.append(
                    f"{c}: {new} allows exceeds baseline {old}; --update-baseline only lowers"
                )
        if errors:
            for e in errors:
                print("FAIL:", e)
            return 1
        new = {c: n for c, n in counts.items() if n > 0}
        write_baseline(root, new)
        print(f"baseline written: {sum(new.values())} allows in {len(new)} crates")
        return 0

    for c in sorted(set(counts) | set(base)):
        n, b = counts.get(c, 0), base.get(c, 0)
        if n > b:
            errors.append(
                f"{c}: {n} #[allow({LINT})] > baseline {b}. Bundle the arguments "
                "into a struct instead of adding an allow."
            )
        elif n < b:
            msg = (
                f"{c}: {n} < baseline {b}. Good: lock it in with "
                "`python3 scripts/too_many_arguments_ratchet.py --update-baseline`."
            )
            if args.strict:
                errors.append(msg)
            else:
                print("NOTE:", msg)
    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1
    print(
        f"too_many_arguments ratchet: OK ({sum(counts.values())} allows in {len(counts)} crates)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
