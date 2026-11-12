#!/usr/bin/env python3
"""Check that each energy anchor's evidence test fails on its committed mutation.

Every entry of ENERGY_ANCHORS (crates/ferric-core/src/certificate/anchors.rs)
has a row in testdata/mutations/ledger.json naming a production-code patch, the
anchor's evidence test file, one test function in it and optional environment
variables. For each selected row this script:

1. refuses a patch that changes no file, or a file outside `crates/*/src/`;
2. creates a scratch worktree of HEAD under --scratch;
3. runs the test unpatched and requires it to pass (skip with --no-baseline);
   every baseline runs before the first patch, so the unpatched build is
   compiled once;
4. applies the patch to a clean tree and runs the same test again, through
   --cargo:

       cargo test -p <crate> --test <stem> -- --exact <test> --include-ignored

5. accepts the mutation only when that run fails on an assertion: the test's
   own thread panics at an `assert!`, `assert_eq!` or `assert_ne!` line of a
   file under the crate's `tests/` directory, and the panic message is not an
   `unwrap()` payload or the literal of an `.expect("...")` call on that line.
   A compile error, any other panic, a crash or a timeout is rejected;
6. removes the scratch worktree.

With --write the result of every accepted row is written back to the ledger:
the patch's sha256, the base commit, the failing test, the assertion location
and the first line of its message. A selected row that is no longer accepted
loses its recorded result, so the ledger test reports it as pending. The ferric-core test
`mutation_ledger_matches_the_registry` checks the ledger against the registry
and the patch files on every run; this script is run on demand:

    box-lease run --label anchor-mutations -- \\
        scripts/check_anchor_mutations.py --scratch DIR [--anchor RHF ...] --write

Exit status 0 when every selected row is accepted, 1 otherwise.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = Path("testdata/mutations/ledger.json")
ASSERT_LINE = re.compile(r"\b(assert|assert_eq|assert_ne)!\s*\(")
# Rust >= 1.89 prints the thread id: "thread 'name' (1234) panicked at".
# Payloads of `Option::unwrap`/`Result::unwrap` panics, and the literal of an
# `.expect("...")` call: a panic carrying one of these on an assertion line
# comes from the unwrap or expect, not from the assertion comparing.
UNWRAP_PANIC = re.compile(r"^called `(Option|Result)::unwrap(_err)?\(\)` on an? `")
EXPECT_CALL = re.compile(r'\.expect\(\s*"((?:[^"\\]|\\.)*)"')
PANIC = re.compile(r"^thread '([^']*)'(?: \(\d+\))? panicked at (\S+?):(\d+):\d+:\s*$")

CAUGHT = "assertion failure"
NOT_SEEN = "not seen to fail"


def patch_paths(text):
    """Every file a unified diff touches: both sides of each `diff --git a/X b/Y`
    header and the `rename`/`copy` from/to lines."""
    paths = []
    for a, b in re.findall(r"^diff --git a/(.+) b/(.+)$", text, re.MULTILINE):
        paths += [a, b]
    paths += re.findall(r"^(?:rename|copy) (?:from|to) (.+)$", text, re.MULTILINE)
    return list(dict.fromkeys(paths))


def production_path(path):
    parts = path.split("/")
    return len(parts) >= 4 and parts[0] == "crates" and parts[2] == "src"


def evidence_target(evidence):
    """(package, test target) of `crates/<pkg>/tests/<stem>.rs`."""
    parts = evidence.split("/")
    if (
        len(parts) != 4
        or parts[0] != "crates"
        or parts[2] != "tests"
        or not parts[3].endswith(".rs")
    ):
        raise ValueError(f"evidence {evidence!r} is not crates/<pkg>/tests/<stem>.rs")
    return parts[1], parts[3][:-3]


def run_test(cargo, wt, row, target_dir, timeout):
    """Run the row's test in `wt`; return (classification, details)."""
    pkg, stem = evidence_target(row["evidence"])
    cmd = [
        cargo,
        "test",
        "-p",
        pkg,
        "--test",
        stem,
        "--",
        "--exact",
        row["test"],
        "--include-ignored",
    ]
    env = dict(os.environ)
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env["CARGO_TARGET_DIR"] = str(target_dir)
    env.update(row.get("env") or {})
    proc = subprocess.Popen(
        cmd,
        cwd=wt,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return "timeout", {"seconds": timeout}
    return classify(out, proc.returncode, row, wt)


def classify(out, rc, row, wt):
    """Classify one cargo test run of a single test (see the module doc)."""
    lines = out.splitlines()
    test = row["test"]
    if re.search(rf"^test {re.escape(test)} \.\.\. ok$", out, re.MULTILINE) and rc == 0:
        return "passed", {}
    if not re.search(r"^running 1 test$", out, re.MULTILINE):
        if "could not compile" in out or re.search(
            r"^error(\[E\d+\])?:", out, re.MULTILINE
        ):
            return "compile error", {"tail": lines[-5:]}
        if re.search(r"^running 0 tests$", out, re.MULTILINE):
            return "test not found", {}
        return "no test run", {"rc": rc, "tail": lines[-5:]}
    if not re.search(rf"^test {re.escape(test)} \.\.\. FAILED$", out, re.MULTILINE):
        return "crashed", {"rc": rc, "tail": lines[-5:]}
    pkg, _ = evidence_target(row["evidence"])
    for i, line in enumerate(lines):
        m = PANIC.match(line)
        if not m:
            continue
        thread, path, lineno = m.group(1), m.group(2), int(m.group(3))
        message = lines[i + 1] if i + 1 < len(lines) else ""
        where = {"thread": thread, "assertion": f"{path}:{lineno}", "message": message}
        if thread != test:
            return "panic outside an assertion", where
        if not path.startswith(f"crates/{pkg}/tests/"):
            return "panic outside an assertion", where
        src = Path(wt, path)
        text = src.read_text().splitlines() if src.is_file() else []
        if not (0 < lineno <= len(text) and ASSERT_LINE.search(text[lineno - 1])):
            return "panic outside an assertion", where
        line_text = text[lineno - 1]
        raised_by_unwrap = UNWRAP_PANIC.match(message) or any(
            message.startswith(m) for m in EXPECT_CALL.findall(line_text)
        )
        if raised_by_unwrap:
            return "panic outside an assertion", where
        return CAUGHT, where
    return "crashed", {"rc": rc, "tail": lines[-5:]}


def git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def precheck(row, repo):
    """The report of a row whose patch is refused before any run, else None."""
    patch = repo / row["patch"]
    row["_sha256"] = hashlib.sha256(patch.read_bytes()).hexdigest()
    paths = patch_paths(patch.read_text())
    if not paths:
        return {"status": NOT_SEEN, "reason": "the patch changes no file"}
    outside = [p for p in paths if not production_path(p)]
    if outside:
        return {"status": "not a production mutation", "reason": f"changes {outside}"}
    return None


def mutate(row, repo, wt, base, args):
    """Apply the row's patch to a clean worktree and run its test."""
    git("reset", "-q", "--hard", base, cwd=wt)
    applied = subprocess.run(
        ["git", "apply", str(repo / row["patch"])],
        cwd=wt,
        capture_output=True,
        text=True,
    )
    if applied.returncode != 0:
        return {"status": "patch does not apply", "reason": applied.stderr.strip()}
    status, details = run_test(args.cargo, wt, row, args.target_dir, args.timeout)
    if status == "passed":
        status = NOT_SEEN
    elif status == "compile error":
        status = "compile error: not a mutation"
    return {"status": status, "baseline_run": not args.no_baseline, **details}


def check_rows(rows, repo, wt, base, args):
    """Report per row. Every baseline runs first, on the unpatched tree, so
    the unpatched build is compiled once; then each patch runs on a reset
    tree."""
    reports = {r["anchor"]: precheck(r, repo) for r in rows}
    live = [r for r in rows if reports[r["anchor"]] is None]
    if not args.no_baseline:
        for row in live:
            status, details = run_test(
                args.cargo, wt, row, args.target_dir, args.timeout
            )
            if status != "passed":
                reports[row["anchor"]] = {"status": f"baseline {status}", **details}
    for row in live:
        if reports[row["anchor"]] is None:
            reports[row["anchor"]] = mutate(row, repo, wt, base, args)
    out = []
    for row in rows:
        r = {
            "anchor": row["anchor"],
            "patch_sha256": row.pop("_sha256"),
            **reports[row["anchor"]],
        }
        print(json.dumps(r), flush=True)
        out.append((row, r))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--repo",
        type=Path,
        default=REPO,
        help="repository root (default: this checkout)",
    )
    p.add_argument(
        "--scratch", type=Path, required=True, help="directory for the scratch worktree"
    )
    p.add_argument(
        "--anchor", action="append", help="anchor method to check (default: every row)"
    )
    p.add_argument("--cargo", default="cargo", help="cargo executable")
    p.add_argument(
        "--target-dir",
        type=Path,
        help="CARGO_TARGET_DIR (default: <scratch>/anchor-mutation-target)",
    )
    p.add_argument(
        "--timeout", type=float, default=1800.0, help="seconds per cargo run"
    )
    p.add_argument("--no-baseline", action="store_true", help="skip the unpatched run")
    p.add_argument(
        "--write", action="store_true", help="record accepted rows in the ledger"
    )
    args = p.parse_args(argv)
    repo = args.repo.resolve()
    args.target_dir = (
        args.target_dir or args.scratch / "anchor-mutation-target"
    ).resolve()
    ledger_path = repo / LEDGER
    ledger = json.loads(ledger_path.read_text())
    rows = [r for r in ledger["rows"] if not args.anchor or r["anchor"] in args.anchor]
    unknown = set(args.anchor or []) - {r["anchor"] for r in rows}
    if unknown:
        p.error(f"no ledger row for {sorted(unknown)}")
    base = git("rev-parse", "HEAD", cwd=repo)
    wt = (args.scratch / "anchor-mutation-worktree").resolve()
    if wt.exists():
        p.error(f"{wt} exists; remove it first")
    args.scratch.mkdir(parents=True, exist_ok=True)
    git("worktree", "add", "-q", "--detach", str(wt), base, cwd=repo)
    try:
        reports = check_rows(rows, repo, wt, base, args)
    finally:
        git("worktree", "remove", "--force", str(wt), cwd=repo)
    if args.write:
        today = datetime.date.today().isoformat()
        for row, r in reports:
            if r["status"] == CAUGHT:
                row["result"] = {
                    "patch_sha256": r["patch_sha256"],
                    "base_commit": base,
                    "failing_test": row["test"],
                    "assertion": r["assertion"],
                    "message": r["message"],
                    "baseline_run": r["baseline_run"],
                    "date": today,
                }
            else:
                # A row that is no longer accepted keeps no earlier result: the
                # ledger test then reports it as pending.
                row["result"] = None
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    bad = [r["anchor"] for _, r in reports if r["status"] != CAUGHT]
    print(
        f"{len(reports) - len(bad)} of {len(reports)} accepted"
        + (f"; not accepted: {bad}" if bad else "")
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
