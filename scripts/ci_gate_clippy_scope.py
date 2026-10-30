#!/usr/bin/env python3
"""Decide what the push gate's clippy step lints: changed crates, all, or nothing.

Run from anywhere inside a checkout. Prints exactly three lines, read by
scripts/ci-gate.sh:

    <mode>     crates | workspace | none
    <crates>   space-separated package names (empty unless mode is crates)
    <reason>   one line, printed in the gate summary

Changed files are the diff from merge-base(origin/main, HEAD) to the WORKING
TREE, plus untracked files that cargo could see. The gate lints the working
tree, so the selection has to describe the working tree too. origin/main is
fetched first; merge-base holds up for rebased, force-pushed and stacked
branches, because it does not depend on what the remote branch used to be.

FAIL-SAFE RULE: anything uncertain selects the workspace. That covers a failed
fetch, no merge-base, a git or cargo metadata failure, and any path that is not
owned by a package and not known to be compile-neutral. NEUTRAL_* below is an
allow-list on purpose, so a new top-level path costs a slower gate rather than
lost coverage.

Compile inputs outside the packages:
  - Cargo.toml, Cargo.lock, .cargo/, clippy.toml, rust-toolchain: by name.
  - pyproject.toml: ferric-build-info/build.rs reads the version from it. Falls
    through to the default, like any path not listed as neutral.
  - testdata/: tests include_str! a few files, so only a DELETED file is treated
    as a compile input. Editing one cannot change a lint, and reference data
    changes in too many PRs to pay for a workspace run each time.
test_ci_gate_clippy_scope.py fails if Rust source starts including from a path
listed as neutral.

--base/--head select a fixed range instead, for dry runs over merged PRs: no
fetch, no working tree. Packages still come from the current checkout.
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import PurePosixPath

# A file with one of these names changes how crates compile or lint, wherever
# it is, so it is checked before package ownership.
COMPILE_CONFIG_NAMES = {
    "Cargo.lock",
    "clippy.toml",
    ".clippy.toml",
    "rust-toolchain",
    "rust-toolchain.toml",
}
COMPILE_CONFIG_DIRS = {".cargo"}

# Everything here was checked on 2026-10-30: no Rust source includes from it and
# no build script reads it. benchmarks/ is data; its one package,
# benchmarks/harness, is matched by ownership first.
NEUTRAL_DIRS = (
    ".github/",
    "benchmarks/",
    "examples/",
    "prototypes/",
    "reference/",
    "scripts/",
    "site/",
    "terf-tables/",
)
NEUTRAL_FILES = {
    ".dockerignore",
    ".git-blame-ignore-revs",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".python-version",
    ".secrets.baseline",
    ".config/nextest.toml",
    "CITATION.cff",
    "Dockerfile",
    "uv.lock",
}
NEUTRAL_GLOBS = ("*.md", "LICENSE-*")

GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
FETCH_TIMEOUT_SECS = 60


@dataclass
class Selection:
    mode: str
    reason: str
    crates: list = field(default_factory=list)


def workspace(reason):
    return Selection("workspace", reason)


def owner_of(path, packages):
    """The package whose directory holds `path`; the deepest wins if nested."""
    best_dir, best_name = "", None
    for pkg_dir, name in packages.items():
        if path.startswith(pkg_dir + "/") and len(pkg_dir) > len(best_dir):
            best_dir, best_name = pkg_dir, name
    return best_name


def is_compile_config(path):
    p = PurePosixPath(path)
    return p.name in COMPILE_CONFIG_NAMES or bool(COMPILE_CONFIG_DIRS & set(p.parts))


def is_neutral(path):
    return (
        path in NEUTRAL_FILES
        or path.startswith(NEUTRAL_DIRS)
        or any(fnmatch(PurePosixPath(path).name, g) for g in NEUTRAL_GLOBS)
    )


def classify(changes, packages):
    """Turn (status, path) pairs and {package dir: name} into a Selection.

    Status is git's letter; with --no-renames a rename is a D plus an A. The
    root manifest is not a package here (the workspace is virtual, and
    load_packages drops a root package), so it reaches the default and selects
    the workspace.
    """
    crates, uncertain = set(), []
    for status, path in changes:
        deleted = status == "D"
        if is_compile_config(path):
            uncertain.append(path)
        elif (name := owner_of(path, packages)) is not None:
            crates.add(name)
        elif path.startswith("testdata/"):
            if deleted:  # tests include_str! some of these
                uncertain.append(path)
        elif is_neutral(path):
            pass
        elif path.startswith("crates/") and deleted:
            # Its package is gone from the workspace, so `-p` would make cargo
            # error. Removing a real member also changes Cargo.lock, which
            # selects the workspace above; this is only the orphan directory.
            pass
        else:
            uncertain.append(path)

    if uncertain:
        more = f" (+{len(uncertain) - 1} more)" if len(uncertain) > 1 else ""
        return workspace(
            f"compile input or unclassified path changed: {uncertain[0]}{more}"
        )
    if crates:
        return Selection("crates", f"{len(changes)} changed file(s)", sorted(crates))
    if changes:
        return Selection("none", f"{len(changes)} changed file(s), no compile inputs")
    return Selection("none", "nothing changed against the base")


def git(*args, timeout=None):
    """stdout of a git command, or None if it failed."""
    try:
        r = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=GIT_ENV,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def load_packages(root):
    """{dir relative to root: package name}, or None if cargo could not say."""
    try:
        r = subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1", "--offline"],
            capture_output=True,
            text=True,
            cwd=root,
            stdin=subprocess.DEVNULL,
        )
        if r.returncode != 0:
            return None
        meta = json.loads(r.stdout)
        packages = {}
        for pkg in meta["packages"]:
            manifest_dir = os.path.dirname(os.path.realpath(pkg["manifest_path"]))
            rel = os.path.relpath(manifest_dir, root)
            if rel != ".":
                packages[rel] = pkg["name"]
        return packages
    except (OSError, ValueError, KeyError):
        return None


def parse_name_status(out):
    fields = out.split("\0")
    return [(fields[i][0], fields[i + 1]) for i in range(0, len(fields) - 1, 2)]


def changed_files(base, head, packages):
    """(status, path) pairs, or None if git failed."""
    target = [head] if head else []
    out = git("diff", "--name-status", "--no-renames", "-z", base, *target)
    if out is None:
        return None
    changes = parse_name_status(out)
    if head:
        return changes
    # Untracked files count only where cargo would see them, so a stray scratch
    # file at the top of a checkout does not force a workspace run.
    untracked = git("ls-files", "--others", "--exclude-standard", "-z")
    if untracked is None:
        return None
    for path in filter(None, untracked.split("\0")):
        if owner_of(path, packages) is not None or is_compile_config(path):
            changes.append(("A", path))
    return changes


def resolve_base():
    """(merge-base, None), or (None, why the base is unknown)."""
    if git("fetch", "--quiet", "origin", "main", timeout=FETCH_TIMEOUT_SECS) is None:
        return None, "could not fetch origin/main"
    out = git("merge-base", "origin/main", "HEAD")
    if out is None:
        return None, "no merge-base with origin/main"
    return out.strip(), None


def select(base=None, head=None):
    top = git("rev-parse", "--show-toplevel")
    if top is None:
        return workspace("not inside a git checkout")
    root = os.path.realpath(top.strip())
    os.chdir(root)
    if base is None:
        base, why = resolve_base()
        if base is None:
            return workspace(why)
    packages = load_packages(root)
    if packages is None:
        return workspace("cargo metadata failed")
    changes = changed_files(base, head, packages)
    if changes is None:
        return workspace(f"git diff failed against {base}")
    return classify(changes, packages)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base", help="fixed base revision; skips fetch and merge-base"
    )
    parser.add_argument(
        "--head", help="fixed head revision (needs --base); ignores the tree"
    )
    args = parser.parse_args()
    if args.head and not args.base:
        parser.error("--head needs --base")
    try:
        result = select(args.base, args.head)
    except Exception as exc:  # any surprise must widen the lint, never drop it
        result = workspace(f"selector error: {type(exc).__name__}")
    print(result.mode)
    print(" ".join(result.crates))
    print(result.reason.replace("\n", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
