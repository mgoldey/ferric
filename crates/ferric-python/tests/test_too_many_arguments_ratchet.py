"""The too_many_arguments ratchet only lets the per-crate allow count go down.

`scripts/too_many_arguments_ratchet.py` counts item-level
`#[allow(clippy::too_many_arguments)]` per crate against a checked-in
baseline; `scripts/check_testkit_dev_only.py` keeps `ferric-testkit` a
dev-dependency. Both run here against throwaway trees, plus once against the
real checkout, so the committed baseline cannot drift from the source.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RATCHET = REPO / "scripts" / "too_many_arguments_ratchet.py"
DEV_ONLY = REPO / "scripts" / "check_testkit_dev_only.py"

ALLOW = "#[allow(clippy::too_many_arguments)]\n"
FN = "fn f(a: i32) {}\n"


def run(script, root, *args):
    return subprocess.run(
        [sys.executable, str(script), "--root", str(root), *args],
        capture_output=True,
        text=True,
    )


def tree(tmp_path, counts):
    """A tree with `counts[crate]` item-level allows in each crate."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "Cargo.toml").write_text("[workspace.lints.clippy]\n")
    for crate, n in counts.items():
        src = tmp_path / "crates" / crate / "src"
        src.mkdir(parents=True)
        (src / "lib.rs").write_text((ALLOW + FN) * n)
    return tmp_path


def baseline(root):
    return json.loads(
        (root / "scripts" / "too_many_arguments_baseline.json").read_text()
    )


def test_real_checkout_matches_baseline():
    r = run(RATCHET, REPO)
    assert r.returncode == 0, r.stdout + r.stderr


def test_real_checkout_has_no_crate_level_allow_and_no_workspace_allow():
    # The compiler half: the workspace table must not switch the lint off.
    text = (REPO / "Cargo.toml").read_text()
    assert 'too_many_arguments = "allow"' not in text


def test_new_allow_over_baseline_fails(tmp_path):
    root = tree(tmp_path, {"a": 2, "b": 1})
    assert run(RATCHET, root, "--update-baseline").returncode == 0
    assert baseline(root)["crates"] == {"a": 2, "b": 1}
    assert run(RATCHET, root).returncode == 0
    (root / "crates/a/src/lib.rs").write_text((ALLOW + FN) * 3)
    r = run(RATCHET, root)
    assert r.returncode == 1
    assert "a: 3" in r.stdout and "baseline 2" in r.stdout


def test_allow_in_a_new_crate_fails(tmp_path):
    root = tree(tmp_path, {"a": 1})
    run(RATCHET, root, "--update-baseline")
    src = root / "crates" / "c" / "src"
    src.mkdir(parents=True)
    (src / "lib.rs").write_text(ALLOW + FN)
    r = run(RATCHET, root)
    assert r.returncode == 1 and "c: 1" in r.stdout


def test_lowering_is_a_notice_not_a_failure(tmp_path):
    # Two PRs that each lower the same crate merge cleanly to a baseline that
    # is now too high; main must not go red for that.
    root = tree(tmp_path, {"a": 3})
    run(RATCHET, root, "--update-baseline")
    (root / "crates/a/src/lib.rs").write_text((ALLOW + FN) * 1)
    r = run(RATCHET, root)
    assert r.returncode == 0
    assert "NOTE" in r.stdout and "--update-baseline" in r.stdout
    assert run(RATCHET, root, "--strict").returncode == 1
    assert run(RATCHET, root, "--update-baseline").returncode == 0
    assert baseline(root)["crates"] == {"a": 1}
    assert "total" not in baseline(root)
    assert run(RATCHET, root, "--strict").returncode == 0


def test_update_baseline_never_raises(tmp_path):
    root = tree(tmp_path, {"a": 1})
    run(RATCHET, root, "--update-baseline")
    (root / "crates/a/src/lib.rs").write_text((ALLOW + FN) * 2)
    r = run(RATCHET, root, "--update-baseline")
    assert r.returncode == 1
    assert baseline(root)["crates"] == {"a": 1}


def test_crate_level_allow_is_always_an_error(tmp_path):
    root = tree(tmp_path, {"a": 1})
    run(RATCHET, root, "--update-baseline")
    lib = root / "crates/a/src/lib.rs"
    lib.write_text("#![allow(clippy::too_many_arguments)]\n" + lib.read_text())
    r = run(RATCHET, root)
    assert r.returncode == 1 and "crate-level" in r.stdout


def test_multi_lint_and_multiline_allows_are_counted(tmp_path):
    root = tree(tmp_path, {"a": 0})
    (root / "crates/a/src/lib.rs").write_text(
        "#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]\n"
        + FN
        + "#[allow(\n    clippy::type_complexity,\n    clippy::too_many_arguments,\n)]\n"
        + FN
        + "// #[allow(clippy::too_many_arguments)] in a comment does not count\n"
        + "#[allow(clippy::type_complexity)]\n"
        + FN
    )
    run(RATCHET, root, "--update-baseline")
    assert baseline(root)["crates"] == {"a": 2}


def test_cfg_attr_allow_is_counted(tmp_path):
    root = tree(tmp_path, {"a": 0})
    (root / "crates/a/src/lib.rs").write_text(
        '#[cfg_attr(feature = "x", allow(clippy::too_many_arguments))]\n'
        + FN
        + "#[expect(clippy::too_many_arguments)]\n"
        + FN
    )
    run(RATCHET, root, "--update-baseline")
    assert baseline(root)["crates"] == {"a": 2}


@pytest.mark.parametrize(
    "attr",
    [
        "#[allow(clippy::complexity)]",
        "#![allow(clippy::all)]",
        "#[allow(dead_code, clippy::all)]",
    ],
)
def test_group_allows_are_errors(tmp_path, attr):
    root = tree(tmp_path, {"a": 1})
    run(RATCHET, root, "--update-baseline")
    lib = root / "crates/a/src/lib.rs"
    lib.write_text(attr + "\n" + lib.read_text())
    r = run(RATCHET, root)
    assert r.returncode == 1 and "group" in r.stdout


def test_workspace_allow_in_manifest_is_an_error(tmp_path):
    root = tree(tmp_path, {"a": 1})
    run(RATCHET, root, "--update-baseline")
    (root / "Cargo.toml").write_text(
        '[workspace.lints.clippy]\ntoo_many_arguments = "allow"\n'
    )
    r = run(RATCHET, root)
    assert r.returncode == 1 and "manifest" in r.stdout


def manifest(root, crate, body):
    d = root / "crates" / crate
    d.mkdir(parents=True, exist_ok=True)
    (d / "Cargo.toml").write_text(f'[package]\nname = "{crate}"\n' + body)


def test_testkit_only_as_dev_dependency(tmp_path):
    manifest(
        tmp_path,
        "good",
        '[dev-dependencies]\nferric-testkit = { path = "../ferric-testkit" }\n',
    )
    manifest(tmp_path, "ferric-testkit", "[dependencies]\n")
    assert run(DEV_ONLY, tmp_path).returncode == 0


@pytest.mark.parametrize(
    "body",
    [
        '[dependencies]\nferric-testkit = { path = "../ferric-testkit" }\n',
        '[build-dependencies]\nferric-testkit = { path = "../ferric-testkit" }\n',
        '[dependencies.ferric-testkit]\npath = "../ferric-testkit"\n',
        "[target.'cfg(unix)'.dependencies]\nferric-testkit = { path = \"../ferric-testkit\" }\n",
    ],
)
def test_testkit_as_normal_dependency_fails(tmp_path, body):
    manifest(tmp_path, "bad", body)
    r = run(DEV_ONLY, tmp_path)
    assert r.returncode == 1 and "bad" in r.stdout


def test_real_checkout_testkit_is_dev_only():
    r = run(DEV_ONLY, REPO)
    assert r.returncode == 0, r.stdout
