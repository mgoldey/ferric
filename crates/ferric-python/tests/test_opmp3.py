"""`ferric.run_opmp3`: OP-MP3 through the binding.

CH2 3B1 / cc-pVDZ, ROHF reference, all electrons, exact integrals, against
the paper's Q-Chem 6.4 values (Zenodo 10.5281/zenodo.23482348,
`testdata/reference/opmp3/qchem_cc-pvdz.json`). The construction is anchored
in `crates/ferric-mp2/tests/opmp3*.rs`; these checks cover the binding: the
reference-path numbers match Q-Chem, the W traces are the MP2 and MP3
energies, `total` adds `E_HF + E_NBS`, the certificate names the kind and
its Smoke grade, and bad arguments raise.

Bar: `crates/ferric-mp2/tests/opmp3_qchem.rs` derives it, half a unit of the
last printed Q-Chem digit (1e-10 printed here) plus `10 |E_c| (1e-9 + 1e-10)`
for the two SCF thresholds, 2e-9 Ha in all.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import ferric

ROOT = Path(__file__).resolve().parents[3]
QCHEM = json.loads((ROOT / "testdata/reference/opmp3/qchem_cc-pvdz.json").read_text())
BAR = 2e-9


@pytest.fixture(scope="module")
def res():
    entry = QCHEM["molecules"]["ch2"]
    mol = ferric.Molecule.from_xyz_string(entry["xyz"], 0, 3)
    r = ferric.run_opmp3(
        mol,
        ferric.BasisSet.bundled("cc-pvdz"),
        scf_energy_conv=1e-12,
        scf_density_conv=1e-10,
    )
    return r, entry["hf"]


def _trace(ws, channel):
    return sum(np.trace(ws[f"{s}_{channel}"]) for s in ("alpha", "beta"))


def test_reference_path_matches_qchem(res):
    r, q = res
    w2 = r.w(2)
    got = {
        "e_hf": r.e_hf,
        "e_nbs": r.e_nbs,
        "tr_w2_ss": _trace(w2, "ss"),
        "tr_w2_os": _trace(w2, "os"),
        "tr_w3": _trace(r.w(3), "ss") + _trace(r.w(3), "os"),
    }
    for k, v in got.items():
        assert abs(v - q[k]) < BAR, (k, v, q[k])


def test_traces_are_the_series_energies_and_totals_add_the_reference(res):
    r, _ = res
    c = r.correlation
    e2 = _trace(r.w(2), "ss") + _trace(r.w(2), "os")
    e3 = _trace(r.w(3), "ss") + _trace(r.w(3), "os")
    assert abs(c["mp2"] - e2) < 1e-12
    assert abs(c["mp3"] - (e2 + e3)) < 1e-12
    assert r.x_ss == 0.0
    for k, e in c.items():
        assert r.total(k) == pytest.approx(r.e_hf + r.e_nbs + e, abs=1e-12)
    assert r.max_ov[0] > 1e-4 and r.max_ov[1] > 1e-4  # ROHF is not Brillouin


def test_certificate(res):
    r, _ = res
    cert = r.certificate
    assert cert["capability"] == "opmp3"
    assert cert["grade"] == "smoke"
    assert any(a["name"] == "ssd-opmp3" for a in cert["approximations"])


def test_bad_arguments_raise(res):
    r, _ = res
    with pytest.raises(ValueError):
        r.total("op-mp4")
    with pytest.raises(ValueError):
        r.w(4)
    mol = ferric.Molecule.from_xyz_string("1\n\nHe 0 0 0\n")
    with pytest.raises(Exception, match="x_ss"):
        ferric.run_opmp3(mol, ferric.BasisSet.bundled("cc-pvdz"), reference="b3lyp")
    with pytest.raises(ValueError):
        ferric.run_opmp3(mol, ferric.BasisSet.bundled("cc-pvdz"), reference=" ")
