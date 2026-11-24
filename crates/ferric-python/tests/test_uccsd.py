"""`run_ccsd` / `run_ccsd_t` on an open-shell molecule reach spin-blocked UCCSD.

NH2 (doublet) / cc-pVDZ against the PySCF DF-UCCSD and UCCSD(T) reference in
``testdata/reference/validation/uccsd`` (``scripts/validation/gen_uccsd.py``).
The Rust validation test holds the tight bars; this checks the binding path:
the UHF route is taken, the spin split is returned and sums to the total,
and the frozen-core and ladder arguments reach the library.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

ROOT = Path(__file__).resolve().parents[3]
NH2 = ROOT / "testdata" / "molecules" / "validation" / "nh2.xyz"
WATER = ROOT / "testdata" / "molecules" / "water.xyz"
REF = ROOT / "testdata" / "reference" / "validation" / "uccsd" / "nh2_cc-pvdz.json"

# The binding runs the library at its default CC convergence; the CLI test
# measured 2.3e-9 Ha against a tight library run on OH. 1e-7 Ha leaves room
# for the SCF defaults and stays 100x below a dropped spin block.
TOL = 1e-7

SCF = {"scf_energy_conv": 1e-11, "scf_density_conv": 1e-9}


@pytest.fixture(scope="module")
def nh2():
    return (
        ferric.Molecule.from_xyz(str(NH2), 0, 2),
        ferric.BasisSet.bundled("cc-pvdz"),
        ferric.BasisSet.bundled("cc-pvdz-ri"),
        json.loads(REF.read_text(encoding="utf-8")),
    )


@pytest.fixture(scope="module")
def nh2_ccsd_t(nh2):
    mol, obs, aux, _ = nh2
    return ferric.run_ccsd_t(mol, obs, aux, **SCF)


def test_open_shell_ccsd_t_matches_pyscf(nh2, nh2_ccsd_t):
    ref = nh2[3]["uccsd_fc0"]
    r = nh2_ccsd_t
    assert r.reference == "uhf"
    sc = r.spin_components
    assert abs(r.correlation_energy - ref["e_corr"]) < TOL
    assert abs(r.t_correction - ref["e_t"]) < TOL
    for key in ("e_aa", "e_ab", "e_bb"):
        assert abs(sc[key] - ref[key]) < TOL, key
    for key in ("aaa", "aab", "abb", "bbb"):
        assert abs(sc[f"t_{key}"] - ref[f"e_t_{key}"]) < TOL, key
    assert abs(sc["e_aa"] + sc["e_ab"] + sc["e_bb"] - r.correlation_energy) < 1e-12
    t_sum = sc["t_aaa"] + sc["t_aab"] + sc["t_abb"] + sc["t_bbb"]
    assert abs(t_sum - r.t_correction) < 1e-12


def test_open_shell_ccsd_has_the_same_ccsd_energy(nh2, nh2_ccsd_t):
    mol, obs, aux, _ = nh2
    r = ferric.run_ccsd(mol, obs, aux, **SCF)
    assert r.reference == "uhf"
    assert r.t_correction is None
    assert set(r.spin_components) == {"e_aa", "e_ab", "e_bb"}
    assert abs(r.correlation_energy - nh2_ccsd_t.correlation_energy) < 1e-12


def test_open_shell_frozen_core_reaches_the_library(nh2):
    mol, obs, aux, ref = nh2
    r = ferric.run_ccsd(mol, obs, aux, frozen_core=1, **SCF)
    assert abs(r.correlation_energy - ref["uccsd_fc1"]["e_corr"]) < TOL
    # Negative control: the frozen-core-0 reference must be missed.
    assert abs(r.correlation_energy - ref["uccsd_fc0"]["e_corr"]) > 100 * TOL


def test_open_shell_refuses_an_explicit_ladder(nh2):
    mol, obs, aux, _ = nh2
    with pytest.raises(ValueError, match="ladder"):
        ferric.run_ccsd(mol, obs, aux, ladder="dense", **SCF)


def test_closed_shell_keeps_the_rhf_route():
    mol = ferric.Molecule.from_xyz(str(WATER))
    r = ferric.run_ccsd(
        mol, ferric.BasisSet.bundled("sto-3g"), ferric.BasisSet.bundled("cc-pvdz-ri")
    )
    assert r.reference == "rhf"
    assert r.spin_components is None
