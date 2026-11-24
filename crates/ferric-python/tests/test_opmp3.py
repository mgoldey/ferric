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


# ---------------------------------------------------------------------------
# local="amplitude-threshold": the binding of ferric_mp2::local_opmp3 (the
# construction is anchored in crates/ferric-mp2/tests/local_opmp3.rs).
# ---------------------------------------------------------------------------

PROPANE = ROOT / "testdata/molecules/alkane_3.xyz"


@pytest.mark.parametrize("reference", ["hf", "pbe0"])
def test_local_eps0_equals_the_canonical_run(reference):
    """eps = 0 keeps every amplitude: every model equals the canonical RI run
    (same aux basis, carried in local["canonical"]) to the CG floor."""
    mol = ferric.Molecule.from_xyz(str(PROPANE))
    r = ferric.run_opmp3(
        mol,
        ferric.BasisSet.bundled("6-31g"),
        reference=reference,
        auxbasis=ferric.BasisSet.bundled("cc-pvdz-ri"),
        local="amplitude-threshold",
        eps=0.0,
        compute_reference=True,
    )
    loc = r.local
    assert loc["scheme"] == "amplitude-threshold" and loc["eps"] == 0.0
    assert loc["keep_fraction"] == pytest.approx(1.0, abs=1e-15)
    for k, e in r.correlation.items():
        assert abs(e - loc["canonical"][k]) < 1e-10, (k, e, loc["canonical"][k])
    assert abs(loc["e3"] - loc["e3_2n1"]) < 1e-10
    assert any(
        a["name"] == "amplitude-threshold" for a in r.certificate["approximations"]
    )


def test_local_arguments_follow_the_cli_rules():
    mol = ferric.Molecule.from_xyz(str(PROPANE))
    bs, aux = ferric.BasisSet.bundled("6-31g"), ferric.BasisSet.bundled("cc-pvdz-ri")
    with pytest.raises(ValueError, match="eps"):
        ferric.run_opmp3(mol, bs, auxbasis=aux, local="amplitude-threshold")
    with pytest.raises(ValueError, match="auxbasis"):
        ferric.run_opmp3(mol, bs, local="amplitude-threshold", eps=1e-4)
    with pytest.raises(ValueError, match="none"):
        ferric.run_opmp3(mol, bs, auxbasis=aux, eps=1e-4, lambda_boys=10.0)
    exact = ferric.run_opmp3(
        ferric.Molecule.from_xyz_string("1\n\nHe 0 0 0\n"),
        ferric.BasisSet.bundled("cc-pvdz"),
    )
    assert exact.local is None
