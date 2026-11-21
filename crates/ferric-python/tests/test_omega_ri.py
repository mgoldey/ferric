"""`run_omega_ri`: the omega-RI fit through the Python binding.

Anchors (each a trivial limit or an independent construction):
* omega = 0 is the Coulomb fit: "rimp2" equals `run_rimp2`, "cdd-mp2"
  equals `run_laplace_mp2` at the same n_quad;
* the omega-RI shift E(omega) - E(0) equals PySCF + numpy on the explicit
  matrix A^T V~^-1 V V~^-1 A (testdata/reference/validation/omega_ri/,
  scripts/validation/gen_omega_ri.py) for MP2 and RPA;
* a key the chosen method does not read raises ValueError.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")

ROOT = Path(__file__).resolve().parents[3]
H2O = str(ROOT / "testdata" / "molecules" / "validation" / "h2o.xyz")
REF = json.loads(
    (ROOT / "testdata" / "reference" / "validation" / "omega_ri" / "h2o_cc-pvdz.json").read_text()
)


@pytest.fixture(scope="module")
def sys():
    return (
        ferric.Molecule.from_xyz(H2O),
        ferric.BasisSet.bundled("cc-pvdz"),
        ferric.BasisSet.bundled("cc-pvdz-ri"),
    )


def test_omega_zero_is_the_coulomb_fit(sys):
    r0 = ferric.run_omega_ri(*sys, 0.0)
    ref = ferric.run_rimp2(*sys)
    assert abs(r0.e_corr - ref.mp2_corr) < 1e-9
    c0 = ferric.run_omega_ri(*sys, 0.0, method="cdd-mp2", n_quad=7)
    lap = ferric.run_laplace_mp2(*sys, n_quad=7)
    assert abs(c0.e_corr - lap.mp2_corr) < 1e-9
    assert c0.pair_fraction == 1.0


@pytest.mark.parametrize("method,key", [("rimp2", "mp2_minus_coulomb"), ("rpa", "rpa_minus_coulomb")])
def test_omega_ri_shift_matches_pyscf(sys, method, key):
    e0 = ferric.run_omega_ri(*sys, 0.0, method=method).e_corr
    for blk in REF["omega_ri"]:
        w = blk["omega_bohr_inv"]
        r = ferric.run_omega_ri(*sys, w, method=method)
        assert r.fit.startswith("erfc")
        shift = r.e_corr - e0
        assert abs(shift - blk[key]) < 2e-9 + 1e-4 * abs(blk[key]), (method, w, shift, blk[key])


def test_cdd_sos_and_screens(sys):
    sos = ferric.run_omega_ri(*sys, 0.1, method="cdd-sos-mp2", c_os=1.0)
    assert sos.e_ss is None and abs(sos.e_corr - sos.e_os) < 1e-14
    full = ferric.run_omega_ri(*sys, 0.1, method="cdd-mp2")
    scr = ferric.run_omega_ri(*sys, 0.1, method="cdd-mp2", cdd_schwarz_thresh=1e-9, cdd_pair_thresh=1e-6)
    assert scr.pair_fraction < 1.0 and abs(scr.e_corr - full.e_corr) < 1e-6


@pytest.mark.parametrize(
    "kw",
    [
        {"method": "ccsd"},
        {"method": "rimp2", "n_quad": 5},
        {"method": "rimp2", "cdd_tol": 1e-6},
        {"method": "cdd-sos-mp2", "cdd_pair_thresh": 1e-6},
        {"method": "cdd-mp2", "c_os": 1.3},
    ],
)
def test_unread_keys_are_refused(sys, kw):
    with pytest.raises(ValueError):
        ferric.run_omega_ri(*sys, 0.1, **kw)


def test_negative_omega_is_refused(sys):
    with pytest.raises(Exception):
        ferric.run_omega_ri(*sys, -0.1)
