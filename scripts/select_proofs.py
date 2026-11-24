"""Choose the derivation proofs a pull request has to run (book.yml, job ``html``).

The proof runner ``crates/ferric-python/tests/test_derivation_scripts.py``
executes every script in its ``CASES`` list, unmodified and once per recorded
mutation: about 680 tests, 26 minutes or more. A pull request that edits only
book prose cannot change any of them. This script reads the changed files and
prints one of:

* ``all``: run every case. Printed for any change to the runners, the book
  checker, this script, the workflow, the registry's shape, an unknown file
  under ``scripts/`` or ``testdata/derivations/``, or when anything here
  cannot be determined.
* ``none``: no proof input changed (book prose, ``crates/*/theory.md``, code
  no proof script reads).
* the script file names to run, space-separated: the ``CASES`` scripts that
  changed, the scripts of every registry method or gap entry that changed, and
  the scripts that read a changed file (a script that checks a transcription
  against Rust source or reference JSON names that file in its source).

Scripts in the runner's slow tier (budget ``SLOW_S`` or more, marked
``validation``) are dropped from a list: the book job deselects that tier on
every event (pyproject.toml ``addopts``), so selecting one would collect
nothing. The nightly schedule, tag pushes and workflow_dispatch never call
this for a selection; they run every fast-tier case. The nightly schedule and
workflow_dispatch also run the slow tier, one job per script, from the list
``--slow`` prints. Standard library only: the book job runs with ``--noconftest``
and installs nothing for it.

Usage: ``python3 scripts/select_proofs.py --base origin/main
[--github-output "$GITHUB_OUTPUT"]``, or ``--files LIST`` (one path per line,
``-`` for stdin) to classify a list without git. The reasons go to stderr.
``--slow [--github-output ...]`` prints the slow-tier scripts as a JSON list
(``slow=<list>`` in the output file), for the slow-proof job's matrix; a case
list it cannot read is an error, never an empty list.
"""

import argparse
import ast
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = "crates/ferric-python/tests/test_derivation_scripts.py"
REGISTRY = "testdata/derivations/registry.json"

# A change to any of these can change what every proof run means.
ALL_PATTERNS = (
    ".github/workflows/book.yml",
    "scripts/book_build.py",
    "scripts/select_proofs.py",
    "crates/ferric-python/tests/test_derivation_*.py",
    "crates/ferric-python/tests/test_book_build.py",
    "crates/ferric-python/tests/test_select_proofs.py",
)

# Prose: the book and the generated per-crate indexes. No proof reads them.
PROSE_PATTERNS = ("book/*", "crates/*/theory.md")


class Selection:
    """``mode`` is ``all``, ``none`` or ``some``; ``scripts`` is set for ``some``."""

    def __init__(self, mode, scripts=(), reasons=()):
        self.mode = mode
        self.scripts = sorted(scripts)
        self.reasons = list(reasons)

    def __str__(self):
        return " ".join(self.scripts) if self.mode == "some" else self.mode


def _literal(runner_source, name):
    for node in ast.parse(runner_source).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and getattr(node.targets[0], "id", None) == name
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"no {name} in {RUNNER}")


def read_cases(runner_source):
    """The script names of the runner's ``CASES`` list, parsed, not imported
    (importing it needs pytest and sympy)."""
    return [case[0] for case in _literal(runner_source, "CASES")]


def read_slow(runner_source):
    """The ``CASES`` scripts in the slow (``validation``) tier."""
    slow_s = _literal(runner_source, "SLOW_S")
    return [c[0] for c in _literal(runner_source, "CASES") if c[1] >= slow_s]


def _matches(path, patterns):
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


def _mentions(source, name):
    """``name`` (a file's base name) occurs in ``source`` as a whole token."""
    return re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w-])", source) is not None


def registry_scripts(old, new, cases):
    """The ``CASES`` scripts of every registry entry that changed, or None when
    the registry's shape changed (top-level keys, or the field names used in
    ``methods`` or ``gaps``) or either side is missing."""
    if not isinstance(old, dict) or not isinstance(new, dict):
        return None
    if set(old) != set(new):
        return None
    picked = set()
    for section in ("methods", "gaps"):
        a, b = old.get(section), new.get(section)
        if not isinstance(a, dict) or not isinstance(b, dict):
            return None
        if not all(isinstance(e, dict) for e in [*a.values(), *b.values()]):
            return None
        # A field name that appears or disappears across the section is a
        # schema change.
        if set().union(*a.values()) != set().union(*b.values()):
            return None
        for mid in set(a) | set(b):
            ea, eb = a.get(mid, {}), b.get(mid, {})
            if ea == eb:
                continue
            scripts = [ea.get("scripts", []), eb.get("scripts", [])]
            if not all(isinstance(x, list) for x in scripts):
                return None
            for script in scripts[0] + scripts[1]:
                if Path(script).name in cases:
                    picked.add(Path(script).name)
    return picked


def select(changed, cases, sources, old_registry=None, new_registry=None, slow=()):
    """Classify ``changed`` (repository-relative paths).

    ``cases``: the runner's script names. ``sources``: script name -> its text,
    for the data-dependency rule. ``old_registry`` / ``new_registry``: the
    parsed registry at the merge base and at the head, needed only when the
    registry changed. ``slow``: the slow-tier scripts, never listed.
    """
    if not changed:
        return Selection("all", reasons=["no changed files listed: run all"])
    cases = set(cases)
    picked, reasons = set(), []
    for path in changed:
        if _matches(path, ALL_PATTERNS):
            return Selection("all", reasons=[f"{path}: proof runner or tooling"])
        if path == REGISTRY:
            got = registry_scripts(old_registry, new_registry, cases)
            if got is None:
                return Selection(
                    "all", reasons=[f"{path}: shape changed or unreadable"]
                )
            picked |= got
            reasons.append(f"{path}: changed entries -> {sorted(got) or 'none'}")
            continue
        if path.startswith("testdata/derivations/"):
            return Selection("all", reasons=[f"{path}: unknown derivations file"])
        if path.startswith("scripts/"):
            name = path[len("scripts/") :]
            if name in cases:
                picked.add(name)
                reasons.append(f"{path}: proof script")
                continue
            return Selection(
                "all", reasons=[f"{path}: not a proof script, may be shared"]
            )
        if _matches(path, PROSE_PATTERNS):
            reasons.append(f"{path}: prose")
            continue
        readers = {s for s in cases if _mentions(sources.get(s, ""), Path(path).name)}
        picked |= readers
        reasons.append(f"{path}: read by {sorted(readers) or 'no proof script'}")
    for name in sorted(picked & set(slow)):
        reasons.append(f"{name}: slow tier, deselected in this job; not listed")
    picked -= set(slow)
    if not picked:
        return Selection("none", reasons=reasons)
    return Selection("some", picked, reasons)


def _git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def _registry_at(rev):
    try:
        return json.loads(_git("show", f"{rev}:{REGISTRY}"))
    except (subprocess.CalledProcessError, ValueError):
        return None


def selection_from_git(base):
    """The selection for ``base...HEAD``; ``all`` when git cannot answer."""
    try:
        merge_base = _git("merge-base", base, "HEAD").strip()
        diff = _git("diff", "--name-only", "--no-renames", f"{merge_base}...HEAD")
    except (subprocess.CalledProcessError, OSError) as e:
        return Selection("all", reasons=[f"git could not diff against {base}: {e}"])
    changed = [line for line in diff.splitlines() if line.strip()]
    return selection_for(changed, old_registry=_registry_at(merge_base))


def selection_for(changed, old_registry=None):
    """The selection for ``changed`` against the working tree; ``all`` when the
    runner's case list cannot be read."""
    try:
        runner = (ROOT / RUNNER).read_text(encoding="utf-8")
        cases, slow = read_cases(runner), read_slow(runner)
    except (OSError, ValueError, SyntaxError, TypeError, IndexError) as e:
        return Selection("all", reasons=[f"cannot read the case list: {e}"])
    sources = {}
    for name in cases:
        try:
            sources[name] = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        except OSError:
            return Selection("all", reasons=[f"scripts/{name} is missing"])
    try:
        new_registry = json.loads((ROOT / REGISTRY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        new_registry = None
    return select(changed, cases, sources, old_registry, new_registry, slow)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--base", help="git ref to diff against (base...HEAD)")
    src.add_argument("--files", help="file with one changed path per line, - for stdin")
    src.add_argument(
        "--slow", action="store_true", help="list the slow-tier scripts as JSON"
    )
    ap.add_argument(
        "--github-output",
        help="append mode=... and scripts=... (with --slow: slow=...) to this file",
    )
    args = ap.parse_args(argv)
    if args.slow:
        slow = json.dumps(read_slow((ROOT / RUNNER).read_text(encoding="utf-8")))
        print(slow)
        if args.github_output:
            with open(args.github_output, "a", encoding="utf-8") as out:
                out.write(f"slow={slow}\n")
        return 0
    if args.base:
        sel = selection_from_git(args.base)
    else:
        text = sys.stdin.read() if args.files == "-" else Path(args.files).read_text()
        changed = [line.strip() for line in text.splitlines() if line.strip()]
        base_registry = None
        if REGISTRY in changed:
            base_registry = _registry_at("HEAD")
        sel = selection_for(changed, old_registry=base_registry)
    for reason in sel.reasons:
        print(f"  {reason}", file=sys.stderr)
    print(f"proof selection: {sel.mode}", file=sys.stderr)
    print(sel)
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as out:
            out.write(f"mode={sel.mode}\nscripts={' '.join(sel.scripts)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
