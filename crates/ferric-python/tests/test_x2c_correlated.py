"""relativistic="x2c1e" on the correlated Python drivers (run_rimp2,
run_scs_mp2, run_ccsd, run_ccsd_t), HBr / cc-pVDZ / cc-pVDZ-RI.

These drivers read the reference's orbitals and orbital energies and never
rebuild the core Hamiltonian, so on an X2C reference they are the correlation
treatment of the X2C Hamiltonian. Checked against PySCF on the same
``sfx2c1e`` reference with the same auxiliary basis, all electrons correlated
(``testdata/reference/x2c_scf_sfx2c1e.json``,
``scripts/gen_pyscf_x2c_scf_refs.py``: DFMP2, dfccsd.RCCSD, ccsd_t()).

Bar, as in ``crates/ferric-mp2/tests/x2c_rimp2.rs``: the NONRELATIVISTIC
disagreement of the same two codes on the same system (it carries the shared
fitting, integral and convergence floors) plus the stated convergence
thresholds. The X2C change of each correlation energy is asserted to exceed
its bar, so the comparison can tell an X2C reference from one that silently
stayed nonrelativistic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ferric = pytest.importorskip("ferric")
if not hasattr(ferric.AoIntegrals, "hcore_x2c"):
    pytest.skip(
        "loaded ferric build has no X2C (libint2 without second derivatives)",
        allow_module_level=True,
    )

REPO = Path(__file__).resolve().parents[3]
REFS = json.loads((REPO / "testdata/reference/x2c_scf_sfx2c1e.json").read_text())
HBR = REFS["systems"]["HBr"]
PYSCF = HBR["bases"]["cc-pvdz"]

SCF_CONV = 1e-10
SCF_KW = dict(scf_energy_conv=SCF_CONV, scf_density_conv=1e-8)
# ferric's CCSD stops at |dE| < 1e-9 (CcConfig); PySCF's at the file's cc_conv_tol.
CC_CONV = 1e-9 + REFS["cc_conv_tol"]


def _mol():
    xyz = HBR["xyz_angstrom"]
    return ferric.Molecule.from_xyz_string(f"{len(xyz.splitlines())}\n\n{xyz}\n")


def _bs():
    return ferric.BasisSet.bundled("cc-pvdz")


def _aux():
    return ferric.BasisSet.bundled("cc-pvdz-ri")


def _approximations(result):
    """Names without the "scf " prefix a post-SCF certificate gives its
    reference's entries."""
    return [
        a["name"].removeprefix("scf ") for a in result.certificate["approximations"]
    ]


def _assert_x2c_matches(label, nr, rel, ref_nr, ref_x2c, conv):
    """|rel - ref_x2c| within the nonrelativistic disagreement plus `conv`,
    and the X2C change larger than that bar."""
    d_nr = abs(nr - ref_nr)
    d_x2c = abs(rel - ref_x2c)
    bar = d_nr + conv
    shift = abs(rel - nr)
    print(
        f"{label}: nr {nr:.10f} x2c {rel:.10f} |d_nr| {d_nr:.2e} |d_x2c| {d_x2c:.2e} "
        f"bar {bar:.2e} X2C change {shift:.2e}"
    )
    assert shift > 10 * bar, "the comparison cannot see the X2C reference"
    assert d_x2c <= bar


@pytest.fixture(scope="module")
def rimp2():
    args = (_mol(), _bs(), _aux())
    return (
        ferric.run_rimp2(*args, **SCF_KW),
        ferric.run_rimp2(*args, relativistic="x2c1e", **SCF_KW),
    )


def test_rimp2_on_an_x2c_reference_matches_pyscf_dfmp2(rimp2):
    nr, rel = rimp2
    conv = 2 * (SCF_CONV + REFS["conv_tol"])
    _assert_x2c_matches(
        "RI-MP2 corr",
        nr.mp2_corr,
        rel.mp2_corr,
        PYSCF["mp2"]["e_corr_nr"],
        PYSCF["mp2"]["e_corr_x2c"],
        conv,
    )
    # The reference is the X2C SCF, and the total is its energy plus E_corr.
    assert (
        abs(rel.rhf_energy - PYSCF["e_x2c"])
        <= abs(nr.rhf_energy - PYSCF["e_nr"]) + conv
    )
    assert rel.total_energy == pytest.approx(rel.rhf_energy + rel.mp2_corr, abs=1e-12)
    assert "x2c1e" in _approximations(rel)
    assert "x2c1e" not in _approximations(nr)


def test_scs_mp2_spin_components_are_those_of_the_x2c_rimp2(rimp2):
    """SCS-MP2 shares RI-MP2's reference and integrals: e_os + e_ss is the
    anchored RI-MP2 correlation energy, on either Hamiltonian."""
    args = (_mol(), _bs(), _aux())
    for label, ref, kw in (
        ("nr", rimp2[0], {}),
        ("x2c", rimp2[1], {"relativistic": "x2c1e"}),
    ):
        scs = ferric.run_scs_mp2(*args, **kw, **SCF_KW)
        print(
            f"SCS-MP2 {label}: e_os + e_ss {scs.e_os + scs.e_ss:.12f} mp2 {ref.mp2_corr:.12f}"
        )
        assert scs.rhf_energy == pytest.approx(ref.rhf_energy, abs=2 * SCF_CONV)
        assert scs.e_os + scs.e_ss == pytest.approx(ref.mp2_corr, abs=2 * SCF_CONV)
        assert ("x2c1e" in _approximations(scs)) == bool(kw)


def test_ccsd_and_triples_on_an_x2c_reference_match_pyscf_dfccsd():
    args = (_mol(), _bs(), _aux())
    nr = ferric.run_ccsd_t(*args, **SCF_KW)
    rel = ferric.run_ccsd_t(*args, relativistic="x2c1e", **SCF_KW)
    conv = 2 * (SCF_CONV + REFS["conv_tol"]) + CC_CONV
    _assert_x2c_matches(
        "CCSD corr",
        nr.correlation_energy,
        rel.correlation_energy,
        PYSCF["ccsd"]["e_corr_nr"],
        PYSCF["ccsd"]["e_corr_x2c"],
        conv,
    )
    _assert_x2c_matches(
        "(T)",
        nr.t_correction,
        rel.t_correction,
        PYSCF["ccsd"]["e_t_nr"],
        PYSCF["ccsd"]["e_t_x2c"],
        conv,
    )
    assert "x2c1e" in _approximations(rel)
    # run_ccsd is the same solve without the triples.
    ccsd = ferric.run_ccsd(*args, relativistic="x2c1e", **SCF_KW)
    assert ccsd.correlation_energy == pytest.approx(rel.correlation_energy, abs=CC_CONV)
    assert ccsd.t_correction is None
    assert "x2c1e" in _approximations(ccsd)


def test_x2c_refusals():
    args = (_mol(), _bs(), _aux())
    with pytest.raises(ValueError, match="with_gradient"):
        ferric.run_rimp2(*args, relativistic="x2c1e", with_gradient=True)
    for run in (
        ferric.run_rimp2,
        ferric.run_scs_mp2,
        ferric.run_ccsd,
        ferric.run_ccsd_t,
    ):
        with pytest.raises(ValueError, match="relativistic"):
            run(*args, relativistic="X2C1E")
