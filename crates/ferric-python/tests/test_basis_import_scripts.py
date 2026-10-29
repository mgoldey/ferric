"""The basis-set import scripts must never leave the bundled tree half-written.

Two CodeRabbit findings on PR #596, each pinned by a test that fails on the
pre-fix script:

* ``scripts/import_psi4_jkfit.py`` wrote each set's JSON as soon as it passed
  the PySCF cross-check, so a LATER failure left an earlier JSON on disk with
  no provenance record. The import is now all-or-nothing.
* ``scripts/fetch_bse_bases.py --check`` applied SPHERICAL_OVERRIDE to the
  shipped file in memory before comparing, so a shipped 6-311G file still
  carrying a Cartesian d shell passed. It must now exit nonzero.

No network: BSE is stubbed and the PySCF comparison is monkeypatched.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def _load(name):
    path = SCRIPTS / f"{name}.py"
    assert path.is_file(), f"{path} is missing -- this test selects on it"
    spec = importlib.util.spec_from_file_location(f"_under_test_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def psi4_import(monkeypatch, tmp_path):
    pytest.importorskip("pyscf")
    mod = _load("import_psi4_jkfit")
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps({"sets": []}) + "\n")
    psi4 = tmp_path / "psi4"
    psi4.mkdir()
    for stem in mod.NAMES:
        (psi4 / f"{stem}.gbs").write_text("unused: parse_gbs is patched\n")
    monkeypatch.setattr(mod, "BUNDLED", bundled)
    monkeypatch.setattr(mod, "PROVENANCE", prov)
    one_shell = {1: [(0, ["1.0"], ["1.0"])]}
    monkeypatch.setattr(mod, "parse_gbs", lambda text: one_shell)
    argv = ["import_psi4_jkfit.py", "--psi4-dir", str(psi4), "--sha", "0" * 40]
    monkeypatch.setattr(sys, "argv", argv)
    return mod, bundled, prov


def test_psi4_import_writes_nothing_when_a_later_set_fails(psi4_import, monkeypatch):
    mod, bundled, prov = psi4_import
    stems = list(mod.NAMES)
    assert len(stems) >= 2, "need a set AFTER the first to fail"
    failing = stems[1]
    monkeypatch.setattr(
        mod, "same_as_pyscf", lambda stem, data: ["boom"] if stem == failing else []
    )
    before = prov.read_text()
    assert mod.main() == 1
    assert list(bundled.iterdir()) == [], "a JSON was written despite a failure"
    assert prov.read_text() == before


def test_psi4_import_writes_every_set_and_its_provenance_on_success(
    psi4_import, monkeypatch
):
    """Guard the guard: the all-or-nothing path must still write on success."""
    mod, bundled, prov = psi4_import
    monkeypatch.setattr(mod, "same_as_pyscf", lambda stem, data: [])
    assert mod.main() == 0
    assert sorted(p.name for p in bundled.iterdir()) == sorted(
        f"{s}.json" for s in mod.NAMES
    )
    files = {r["file"] for r in json.loads(prov.read_text())["sets"]}
    assert files == {f"{s}.json" for s in mod.NAMES}


def _el(function_type):
    d = {
        "function_type": function_type,
        "region": "",
        "angular_momentum": [2],
        "exponents": ["0.75"],
        "coefficients": [["1.0"]],
    }
    s = {
        "function_type": "gto",
        "region": "",
        "angular_momentum": [0],
        "exponents": ["1.0"],
        "coefficients": [["1.0"]],
    }
    return {"electron_shells": [s, d]}


@pytest.fixture
def bse_check(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules, "basis_set_exchange", types.ModuleType("basis_set_exchange")
    )
    mod = _load("fetch_bse_bases")
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    prov = tmp_path / "provenance.json"
    monkeypatch.setattr(mod, "BUNDLED", bundled)
    monkeypatch.setattr(mod, "PROVENANCE", prov)
    stem = "6-311g_d_p"
    assert stem in mod.SPHERICAL_OVERRIDE
    monkeypatch.setattr(mod, "SETS", {stem: "6-311G**"})

    def fake_fetch(stem, name):
        fresh = {"elements": {"17": _el("gto_cartesian")}}
        rec = {"file": f"{stem}.json"}
        rec["function_type_override"] = {"elements_changed": mod.force_spherical(fresh)}
        return fresh, rec

    monkeypatch.setattr(mod, "fetch", fake_fetch)
    monkeypatch.setattr(sys, "argv", ["fetch_bse_bases.py", "--check"])

    def ship(function_type):
        doc = {"elements": {"17": _el(function_type)}, "function_types": []}
        path = bundled / f"{stem}.json"
        path.write_text(json.dumps(doc, indent=1) + "\n")
        return path

    return mod, ship, prov


def test_check_flags_a_shipped_cartesian_6311g_d_shell(bse_check):
    mod, ship, prov = bse_check
    path = ship("gto_cartesian")
    before = path.read_text()
    assert mod.main() != 0, "--check passed a file that still needs the override"
    assert path.read_text() == before, "--check wrote to the shipped file"
    assert not prov.exists(), "--check wrote provenance"


def test_check_passes_an_already_spherical_6311g_file(bse_check):
    """Guard the guard: a correct shipped file must still pass --check."""
    mod, ship, _ = bse_check
    ship("gto_spherical")
    assert mod.main() == 0
