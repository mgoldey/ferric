"""`ferric.hirshfeld_charges` on an ECP atom (HI / def2-SVP).

The default `proatom="scf"` solves each free atom in the molecule's basis
(`ferric_scf::properties::scf_proatom_provider`). The free iodine used to be
solved with all 53 electrons in def2-SVP's valence basis, which replaces a
28-electron core with an ECP; the UHF occupied block ran past the basis
dimension and `UhfState::new` panicked ("end <= axis_len"), reaching Python as
`PanicException`. The free atom now carries the basis set's ECP, as the
molecule's SCF does, so the proatom is the 25-electron valence density, the
same kind of density the molecular result holds.

Checks (no magic numbers):
* the call runs and the charges sum to the molecular charge (0), which the
  routine's renormalization to `mol.nelec()` (the valence count, 26) makes
  exact only if the effective nuclear charges are the ECP ones;
* iodine is the more electronegative atom (Pauling 2.66 vs 2.20 for H), so its
  charge is negative and H's positive, and both stay well below 1 e in
  magnitude (a bare-Z pairing would be off by the 28-electron core);
* the same sign and sum on the Slater-proatom path, which never panicked.
  Its ECP-atom proatom is covered by `test_hirshfeld_slater_ecp.py`.

Run:
    OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest \\
        crates/ferric-python/tests/test_hirshfeld_ecp.py -q
"""

from pathlib import Path

import pytest

import ferric

ROOT = Path(__file__).resolve().parents[3]
HI = ROOT / "testdata" / "molecules" / "validation" / "ecp" / "hi.xyz"

# Renormalized to N_e exactly; only float summation error remains.
TOL_SUM = 1e-10
# A polar-covalent diatomic: Hirshfeld charges are a few tenths of e at most.
MAX_ABS_Q = 0.5


@pytest.fixture(scope="module")
def hi_rhf():
    mol = ferric.Molecule.from_xyz(str(HI))
    bs = ferric.BasisSet.bundled("def2-svp")
    rhf = ferric.run_rhf(mol, bs, energy_conv=1e-10, density_conv=1e-8)
    assert rhf.converged
    return mol, bs, rhf


@pytest.mark.parametrize("proatom", ["scf", "slater"])
def test_hi_hirshfeld_charges_on_the_ecp_atom(hi_rhf, proatom, capfd):
    mol, bs, rhf = hi_rhf
    q_h, q_i = ferric.hirshfeld_charges(mol, bs, rhf, proatom=proatom)
    err = capfd.readouterr().err
    print(f"proatom={proatom}: q(H) = {q_h:+.6f}, q(I) = {q_i:+.6f}")
    if proatom == "scf":
        # Both free atoms (H and the ECP iodine) got SCF proatoms.
        assert "fell back to" not in err, err
    assert abs(q_h + q_i) < TOL_SUM
    assert q_i < 0.0 < q_h
    if proatom == "scf":
        assert abs(q_i) < MAX_ABS_Q


def test_mbd_on_the_ecp_atom_runs():
    """MBD@rsSCS builds its free-atom proatoms with the same provider and its
    free-atom volumes with the same kind of free-atom SCF; both used to panic
    on the ECP iodine the same way. MBD lowers the energy (attractive)."""
    mol = ferric.Molecule.from_xyz(str(HI))
    bs = ferric.BasisSet.bundled("def2-svp")
    plain = ferric.run_dft(mol, bs, functional="PBE")
    mbd = ferric.run_dft(mol, bs, functional="PBE", dispersion="mbd")
    assert plain.converged and mbd.converged
    print(f"E(PBE) = {plain.energy:.10f}, E(PBE+MBD) = {mbd.energy:.10f}")
    assert mbd.energy < plain.energy
