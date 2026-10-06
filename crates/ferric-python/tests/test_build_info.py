"""ferric.__version__ / ferric.build_info(): which build is loaded.

The commit-equals-HEAD check only runs when the loaded extension resolves to a
file under THIS checkout's ``target/`` directory, because only then is "the
checkout these tests sit in" the source the build came from. The main
checkout's venv, for instance, loads a symlink into a separate worktree, and a
wheel install copies the ``.so`` into site-packages; neither says anything
about this checkout's HEAD. The Rust test
``ferric-build-info::tests::embedded_commit_is_the_checkout_head`` covers the
same property unconditionally.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

REPO = Path(__file__).resolve().parents[3]


def _loaded_so() -> Path:
    return Path(ferric.ferric.__file__).resolve()


def test_build_info_has_every_field_with_a_definite_type():
    info = ferric.build_info()
    assert set(info) == {"version", "commit", "dirty", "profile", "libint_version"}
    assert info["version"] == ferric.__version__
    assert info["commit"] == "unknown" or re.fullmatch(
        r"[0-9a-f]{40}|[0-9a-f]{64}", info["commit"]
    )
    if info["commit"] == "unknown":
        assert info["dirty"] is None
    else:
        assert isinstance(info["dirty"], bool)
    assert info["profile"] in {"release", "debug"}
    assert info["libint_version"] == "unknown" or re.fullmatch(
        r"\d+\.\d+\.\d+.*", info["libint_version"]
    )


def test_version_is_never_a_pypi_rejected_local_version():
    assert "+" not in ferric.__version__
    assert ferric.__version__ != "unknown"


def test_commit_is_the_head_of_the_checkout_it_was_built_from():
    so = _loaded_so()
    if REPO / "target" not in so.parents:
        pytest.skip(f"loaded extension {so} was not built in this checkout ({REPO})")
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    info = ferric.build_info()
    assert info["commit"] == head, (
        f"the loaded build is from {info['commit']}, but this checkout is at {head}: rebuild"
    )


def test_build_stamp_matches_the_contract_smeltery_reads():
    b = ferric.__build__
    info = ferric.build_info()
    if info["commit"] == "unknown" or info["dirty"] is None:
        assert b is None
        return
    assert set(b) == {"git_sha", "dirty"}
    assert re.fullmatch(r"[0-9a-f]{40}", b["git_sha"])
    assert b["git_sha"] == info["commit"]
    assert type(b["dirty"]) is bool and b["dirty"] == info["dirty"]
