"""scripts/check_anchor_mutations.py accepts only a mutation caught by an assertion.

The runner is run against a one-crate git repository and a fake `cargo` that
answers from the scratch worktree's source, so each verdict follows from the
patch the runner applied:

* `2.0` in lib.rs: the test's own `assert!` fails (the one accepted outcome);
* `COMPILE_ERROR`: rustc-style errors and no test run;
* `UNWRAP`: the test thread panics inside lib.rs;
* `PANIC_IN_TEST`: the test thread panics at a `panic!` line of the test file;
* `PROD_ASSERT`: the test thread panics at an `assert!` line of lib.rs;
* `OTHER_THREAD`: a worker thread panics at the test's `assert!` line;
* `SLEEP`: the run outlives the timeout;
* anything else: the test passes. FAKE_ALWAYS_FAIL makes even that fail.

Panics print the thread id as Rust >= 1.89 does, except the UNWRAP one, which
uses the older form. The fake cargo logs its arguments and environment, so the tests also check
what the runner asked cargo to run.
"""

import hashlib
import os
import json
import subprocess
import sys
from pathlib import Path

import pytest

# The fake `cargo` is an executable `#!` script found on PATH; Windows cannot
# exec a shebang file, so the runner gets no cargo and writes no report.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the fake cargo is a shebang script on PATH"
)

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "check_anchor_mutations.py"
EVIDENCE = "crates/demo/tests/validation_demo.rs"
LIB = "crates/demo/src/lib.rs"
TEST_SRC = """\
#[test]
#[ignore = "validation: demo"]
fn demo_value() {
    let x = demo::value();
    let _ = std::env::var("DEMO").unwrap_or_else(|_| panic!("unset"));
    assert!(x == 1.0, "value moved: {x}");
}
"""
ASSERT_LINE = 6
PANIC_LINE = 5

FAKE_CARGO = f"""\
#!{sys.executable}
import json, os, sys, time
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps({{"argv": sys.argv[1:], "demo": os.environ.get("DEMO"),
                          "target": os.environ.get("CARGO_TARGET_DIR")}}) + "\\n")
src = open("{LIB}").read()
def run(ok, panic=None, thread="demo_value", tid=" (4242)"):
    print("running 1 test")
    if ok:
        print("test demo_value ... ok\\n\\ntest result: ok. 1 passed; 0 failed")
        sys.exit(0)
    path, line, msg = panic
    print("test demo_value ... FAILED\\n\\nfailures:\\n\\n---- demo_value stdout ----\\n")
    print(f"thread '{{thread}}'{{tid}} panicked at {{path}}:{{line}}:5:\\n{{msg}}")
    print("\\nfailures:\\n    demo_value\\n\\ntest result: FAILED. 0 passed; 1 failed")
    sys.exit(101)
if "COMPILE_ERROR" in src:
    print("error[E0425]: cannot find value `COMPILE_ERROR` in this scope")
    print("error: could not compile `demo` (lib) due to 1 previous error")
    sys.exit(101)
if "SLEEP" in src:
    time.sleep(30)
if "UNWRAP" in src:
    run(False, ("{LIB}", 2, "called `Option::unwrap()` on a `None` value"), tid="")
if "PROD_ASSERT" in src:
    run(False, ("{LIB}", 2, "PROD_ASSERT"))
if "OTHER_THREAD" in src:
    run(False, ("{EVIDENCE}", {ASSERT_LINE}, "value moved: 2"), thread="<unnamed>")
if "PANIC_IN_TEST" in src:
    run(False, ("{EVIDENCE}", {PANIC_LINE}, "unset"))
if "2.0" in src or os.environ.get("FAKE_ALWAYS_FAIL"):
    run(False, ("{EVIDENCE}", {ASSERT_LINE}, "value moved: 2"))
run(True)
"""


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A committed one-crate repository with an empty ledger."""
    r = tmp_path / "repo"
    (r / "crates/demo/src").mkdir(parents=True)
    (r / "crates/demo/tests").mkdir()
    (r / "testdata/mutations").mkdir(parents=True)
    (r / LIB).write_text("pub fn value() -> f64 {\n    1.0\n}\n")
    (r / EVIDENCE).write_text(TEST_SRC)
    _git(tmp_path, "init", "-q", str(r))
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "init")
    cargo = tmp_path / "cargo"
    cargo.write_text(FAKE_CARGO)
    cargo.chmod(0o755)
    return r


def _patch(repo, name, path, old, new):
    """Write testdata/mutations/<name>.patch replacing `old` by `new` in `path`."""
    f = repo / path
    text = f.read_text()
    f.write_text(text.replace(old, new))
    diff = _git(repo, "diff", "--", path) + "\n"
    f.write_text(text)
    out = repo / "testdata/mutations" / f"{name}.patch"
    out.write_text(diff if new != old else "")
    return out


def _run(repo, patch, *extra, env=None, test="demo_value", result=None):
    row = {
        "anchor": "DEMO",
        "patch": str(patch.relative_to(repo)),
        "evidence": EVIDENCE,
        "test": test,
        "env": {"DEMO": "1"},
        "result": result,
    }
    ledger = repo / "testdata/mutations/ledger.json"
    ledger.write_text(json.dumps({"rows": [row]}))
    scratch = repo.parent / "scratch"
    log = repo.parent / "cargo.log"
    full_env = {**os.environ, "FAKE_LOG": str(log), **(env or {})}
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            str(repo),
            "--scratch",
            str(scratch),
            "--cargo",
            str(repo.parent / "cargo"),
            *extra,
        ],
        capture_output=True,
        text=True,
        env=full_env,
    )
    assert not (scratch / "anchor-mutation-worktree").exists(), (
        "scratch worktree left behind"
    )
    assert "anchor-mutation-worktree" not in _git(repo, "worktree", "list")
    reports = [
        json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")
    ]
    calls = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    return proc, reports, calls, json.loads(ledger.read_text())


def test_empty_patch_is_not_seen_to_fail_even_when_the_test_fails(repo):
    patch = _patch(repo, "empty", LIB, "1.0", "1.0")
    proc, (rep,), calls, _ = _run(repo, patch, env={"FAKE_ALWAYS_FAIL": "1"})
    assert rep["status"] == "not seen to fail", rep
    assert calls == [], "an empty patch must not reach cargo"
    assert proc.returncode == 1


def test_a_mutant_the_test_survives_is_not_seen_to_fail(repo):
    patch = _patch(repo, "comment", LIB, "    1.0\n", "    // no-op\n    1.0\n")
    proc, (rep,), calls, ledger = _run(repo, patch, "--write")
    assert rep["status"] == "not seen to fail", rep
    assert len(calls) == 2, "baseline and patched run"
    assert ledger["rows"][0]["result"] is None
    assert proc.returncode == 1


def test_write_drops_the_result_of_a_row_that_is_no_longer_accepted(repo):
    stale = {"patch_sha256": "0" * 64, "base_commit": "old", "date": "2000-01-01"}
    patch = _patch(repo, "comment", LIB, "    1.0\n", "    // no-op\n    1.0\n")
    proc, (rep,), _, ledger = _run(repo, patch, "--write", result=stale)
    assert rep["status"] == "not seen to fail", rep
    assert ledger["rows"][0]["result"] is None
    assert proc.returncode == 1


def test_a_compile_breaking_patch_is_rejected(repo):
    patch = _patch(repo, "broken", LIB, "1.0", "COMPILE_ERROR")
    proc, (rep,), _, ledger = _run(repo, patch, "--write")
    assert rep["status"] == "compile error: not a mutation", rep
    assert ledger["rows"][0]["result"] is None
    assert proc.returncode == 1


def test_an_assertion_failure_is_accepted_and_recorded(repo):
    patch = _patch(repo, "two", LIB, "1.0", "2.0")
    proc, (rep,), calls, ledger = _run(repo, patch, "--write")
    assert rep["status"] == "assertion failure", rep
    assert proc.returncode == 0, proc.stdout + proc.stderr
    want_argv = [
        "test",
        "-p",
        "demo",
        "--test",
        "validation_demo",
        "--",
        "--exact",
        "demo_value",
        "--include-ignored",
    ]
    assert [c["argv"] for c in calls] == [want_argv, want_argv]
    assert all(
        c["demo"] == "1" and c["target"].endswith("anchor-mutation-target")
        for c in calls
    )
    res = ledger["rows"][0]["result"]
    assert res["patch_sha256"] == hashlib.sha256(patch.read_bytes()).hexdigest()
    assert res["base_commit"] == _git(repo, "rev-parse", "HEAD")
    assert res["failing_test"] == "demo_value"
    assert res["assertion"] == f"{EVIDENCE}:{ASSERT_LINE}"
    assert res["message"] == "value moved: 2"
    assert res["baseline_run"] is True


def test_a_panic_in_production_code_is_rejected(repo):
    patch = _patch(repo, "unwrap", LIB, "1.0", "UNWRAP")
    _, (rep,), _, _ = _run(repo, patch)
    assert rep["status"] == "panic outside an assertion", rep
    assert rep["assertion"] == f"{LIB}:2"


def test_a_panic_at_a_non_assert_line_of_the_test_is_rejected(repo):
    patch = _patch(repo, "testpanic", LIB, "1.0", "PANIC_IN_TEST")
    _, (rep,), _, _ = _run(repo, patch)
    assert rep["status"] == "panic outside an assertion", rep


def test_a_timeout_is_rejected(repo):
    patch = _patch(repo, "sleep", LIB, "1.0", "SLEEP")
    _, (rep,), _, _ = _run(repo, patch, "--timeout", "2", "--no-baseline")
    assert rep["status"] == "timeout", rep


def test_a_patch_to_the_test_is_not_a_production_mutation(repo):
    patch = _patch(repo, "edit_test", EVIDENCE, "x == 1.0", "x == 2.0")
    _, (rep,), calls, _ = _run(repo, patch)
    assert rep["status"] == "not a production mutation", rep
    assert calls == []


def test_a_rename_out_of_production_source_is_not_a_production_mutation(repo):
    patch = repo / "testdata/mutations/rename.patch"
    patch.write_text(
        "diff --git a/crates/demo/src/helper.rs b/crates/demo/tests/helper.rs\n"
        "similarity index 100%\n"
        "rename from crates/demo/src/helper.rs\n"
        "rename to crates/demo/tests/helper.rs\n"
    )
    _, (rep,), calls, _ = _run(repo, patch)
    assert rep["status"] == "not a production mutation", rep
    assert "crates/demo/tests/helper.rs" in rep["reason"], rep
    assert "crates/demo/src/helper.rs" not in rep["reason"], rep
    assert calls == []


def test_a_failing_baseline_is_rejected(repo):
    patch = _patch(repo, "two", LIB, "1.0", "2.0")
    _, (rep,), calls, ledger = _run(
        repo, patch, "--write", env={"FAKE_ALWAYS_FAIL": "1"}
    )
    assert rep["status"] == "baseline assertion failure", rep
    assert len(calls) == 1
    assert ledger["rows"][0]["result"] is None


def test_an_assert_in_production_code_is_rejected(repo):
    patch = _patch(
        repo, "prod_assert", LIB, "1.0", '{ assert!(false, "PROD_ASSERT"); 1.0 }'
    )
    _, (rep,), _, _ = _run(repo, patch)
    assert rep["status"] == "panic outside an assertion", rep


def test_a_panic_on_another_thread_is_rejected(repo):
    patch = _patch(repo, "thread", LIB, "1.0", "OTHER_THREAD")
    _, (rep,), _, _ = _run(repo, patch)
    assert rep["status"] == "panic outside an assertion", rep
