#!/usr/bin/env python3
"""Decide which crates' TIERED tests a CI run must execute, from the changed files.

Tiered tests are the ones marked `#[ignore = "tier <name>: CI runs it when an
affected crate changes (scripts/ci_test_tiers.py)"]`; that script lists them. Every other #[ignore] keeps its old
meaning and is never selected here. The per-commit shards skip tiered tests
like any ignored test; this script picks the subset that a change could break,
at crate granularity:

    scripts/ci_test_scope.py --base <rev> [--head <rev>]

prints ONE nextest filterset on stdout and the reason on stderr:

    rdeps(ferric-pbc) | rdeps(ferric-cli)   the changed packages and every
                                            package that depends on them
    all()                                   uncertain: run every tier
    none()                                  nothing a test can read changed

The plumbing (git, cargo metadata, package ownership) is shared with
scripts/ci_gate_clippy_scope.py. The classification differs: a file that
cannot change a lint can still change a test outcome, because tests read
testdata/, examples/, site/src, scripts/, .github/ and top-level files at run
time. So a changed path OUTSIDE every package selects the packages whose Rust
sources (tests, src, build.rs) READ it, found at selection time from their
string literals (comment lines skipped). A package reads path P when one of
its literals, with leading ./ and ../ removed:
  - contains P's file name (include_str!("../../README.md"), "water_rhf.json");
  - IS one of P's ancestor directories ("examples", "site/src/", .join("site"));
  - formats a name inside an ancestor ("testdata/reference/{mol}_{basis}.json",
    "scripts/*.sh").
A path no package reads selects nothing. Residual risk: a test that walks a
directory it reaches without naming it (e.g. the repo root, recursively) is
not seen; the nightly all() run is the backstop for that.

FAIL-SAFE RULE, same as the clippy scope: anything uncertain selects all().
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import PurePosixPath

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_gate_clippy_scope as scope  # noqa: E402

LITERAL = re.compile(r'"((?:[^"\\\n]|\\.)*)"')


def package_literals(root, packages):
    """{package name: set of normalised string literals in its .rs files}."""
    out = defaultdict(set)
    listed = scope.git("ls-files", "-z", "--", "*.rs")
    if listed is None:
        return None
    for rel in filter(None, listed.split("\0")):
        name = scope.owner_of(rel, packages)
        if name is None:
            continue
        try:
            text = open(os.path.join(root, rel), encoding="utf-8").read()
        except OSError:
            return None
        for line in text.split("\n"):
            if line.lstrip().startswith("//"):
                continue
            for lit in LITERAL.findall(line):
                out[name].add(re.sub(r"^(\.\.?/)+", "", lit))
    return out


def readers(path, literals):
    """The packages whose literals read `path` (see the module doc)."""
    p = PurePosixPath(path)
    ancestors = [str(a) for a in p.parents if str(a) != "."]
    hit = set()
    for name, lits in literals.items():
        for lit in lits:
            bare = lit.rstrip("/")
            if (
                p.name in lit
                or bare in ancestors
                or any(f"{a}/{{" in lit or f"{a}/*" in lit for a in ancestors)
            ):
                hit.add(name)
                break
    return hit


def classify(changes, packages, literals):
    """(filterset, reason) for (status, path) pairs and {package dir: name}."""
    crates, uncertain, read_by = set(), [], {}
    for status, path in changes:
        if scope.is_compile_config(path):
            uncertain.append(path)
        elif status == "D" and PurePosixPath(path).name == "Cargo.toml":
            uncertain.append(path)
        elif (name := scope.owner_of(path, packages)) is not None:
            crates.add(name)
        elif path == "Cargo.toml":
            uncertain.append(path)
        else:
            r = readers(path, literals)
            read_by[path] = r
            crates |= r
    if uncertain:
        more = f" (+{len(uncertain) - 1} more)" if len(uncertain) > 1 else ""
        return "all()", f"compile input changed: {uncertain[0]}{more}"
    outside = sum(1 for r in read_by.values() if r)
    note = (
        f"; {outside}/{len(read_by)} outside path(s) read by a test" if read_by else ""
    )
    if crates:
        names = sorted(crates)
        return " | ".join(
            f"rdeps({n})" for n in names
        ), f"changed: {' '.join(names)}{note}"
    return "none()", f"{len(changes)} changed file(s), none readable by a test{note}"


def select(base, head):
    top = scope.git("rev-parse", "--show-toplevel")
    if top is None:
        return "all()", "not inside a git checkout"
    root = os.path.realpath(top.strip())
    os.chdir(root)
    if not base or set(base) == {"0"}:
        return "all()", "no base revision (new branch or first push)"
    if scope.git("cat-file", "-e", f"{base}^{{commit}}") is None:
        return "all()", f"base {base} is not in this clone"
    packages = scope.load_packages(root)
    if packages is None:
        return "all()", "cargo metadata failed"
    changes = scope.changed_files(base, head or "HEAD", packages)
    if changes is None:
        return "all()", f"git diff failed against {base}"
    literals = package_literals(root, packages)
    if literals is None:
        return "all()", "could not read package sources"
    return classify(changes, packages, literals)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="base revision ('' = unknown)")
    parser.add_argument("--head", help="head revision (default HEAD; tree ignored)")
    args = parser.parse_args()
    try:
        expr, reason = select(args.base, args.head)
    except Exception as exc:  # any surprise must widen the run, never drop it
        expr, reason = "all()", f"selector error: {type(exc).__name__}"
    print(f"test scope: {expr}  ({reason})", file=sys.stderr)
    print(expr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
