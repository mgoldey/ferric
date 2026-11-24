"""The Slater fallback proatom of an ECP atom is its valence proatom.

With `proatom="slater"` (and for any atom whose free-atom SCF fails) the
Hirshfeld routines use a model proatom. For an all-electron atom that is the
single exponential Z xi^3/pi exp(-2 xi r), unchanged. An ECP basis holds only
the valence density, so the ECP atom's proatom used to carry Z electrons
(53 for iodine) against a 25-electron density and inflated its share:
q(I) = -0.67 e in HI and -1.56 e in CH3I, vs -0.07 and -0.10 e with the SCF
proatom. It is now built from Slater's-rules STO shells with the n_core
innermost electrons removed (`ferric_scf::properties::SlaterProatom`), so it
integrates to Z - n_core and has a valence shape (it tracks the free iodine's
def2-SVP valence density to within ~30% from 0.5 to 4 Bohr; the
single exponential is off by up to 7x there).

Bound on the iodine charge vs the SCF-proatom charge: 0.3 e. The partner
atoms keep their crude single exponentials (H's is far too compact past
1.5 Bohr, C's 2-8x too diffuse at 2-4 Bohr), which is where the remaining
difference comes from; the single-Slater model already differs from SCF
proatom charges by 0.23-0.72 e on all-electron molecules
(`crates/ferric-rpa/tests/validation_hirshfeld.rs`). Measured: 0.28 e (HI),
0.21 e (CH3I); before the fix 0.60 and 1.46 e. The sign of q(I) is asserted
for HI only: in CH3I the over-diffuse carbon proatom pulls charge from the
iodine (q(I) = +0.10 e), a property of the all-electron model this change
does not touch.

Run:
    OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest \\
        crates/ferric-python/tests/test_hirshfeld_slater_ecp.py -q
"""

from pathlib import Path

import numpy as np
import pytest

import ferric

ROOT = Path(__file__).resolve().parents[3]
ECP = ROOT / "testdata" / "molecules" / "validation" / "ecp"

TOL_SUM = 1e-10
MAX_DQ_IODINE = 0.3


def _rhf(name):
    mol = ferric.Molecule.from_xyz(str(ECP / f"{name}.xyz"))
    bs = ferric.BasisSet.bundled("def2-svp")
    rhf = ferric.run_rhf(mol, bs, energy_conv=1e-10, density_conv=1e-8)
    assert rhf.converged
    return mol, bs, rhf


@pytest.mark.parametrize("name", ["hi", "ch3i"])
def test_slater_charges_of_an_ecp_molecule(name):
    mol, bs, rhf = _rhf(name)
    q_slater = np.asarray(ferric.hirshfeld_charges(mol, bs, rhf, proatom="slater"))
    q_scf = np.asarray(ferric.hirshfeld_charges(mol, bs, rhf))
    print(f"{name}: slater {np.round(q_slater, 4)}  scf {np.round(q_scf, 4)}")
    assert abs(q_slater.sum()) < TOL_SUM
    # Atom order in the validation geometries: HI is (H, I); CH3I is
    # (C, I, H, H, H).
    i_idx = 1
    h_idx = [0] if name == "hi" else [2, 3, 4]
    assert all(q_slater[h] > 0.0 for h in h_idx)
    if name == "hi":
        assert q_slater[i_idx] < 0.0
    dq = abs(q_slater[i_idx] - q_scf[i_idx])
    print(f"{name}: |q_I(slater) - q_I(scf)| = {dq:.3f} e")
    assert dq < MAX_DQ_IODINE


def test_hirshfeld_polarizability_of_an_ecp_molecule():
    """Uses the Slater proatom throughout; finite and physical on HI."""
    mol, bs, rhf = _rhf("hi")
    aux = ferric.BasisSet.bundled("def2-svp-rifit")
    alpha = np.asarray(ferric.hirshfeld_polarizability(mol, bs, aux, rhf))
    print("diag H", np.diag(alpha[0]), "diag I", np.diag(alpha[1]))
    assert alpha.shape == (2, 3, 3)
    assert np.all(np.isfinite(alpha))
    for a in alpha:
        assert np.allclose(a, a.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(0.5 * (a + a.T)) > 0.0)
    # Iodine is far more polarizable than hydrogen.
    assert np.trace(alpha[1]) > np.trace(alpha[0])
