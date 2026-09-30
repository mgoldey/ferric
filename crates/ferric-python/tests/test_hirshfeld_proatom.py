"""`ferric.hirshfeld_charges` proatom selection.

The binding's default (`proatom="scf"`) builds its proatoms with
`ferric_scf::properties::scf_proatom_provider` -- the same function ferric-cli
passes for `[rpa] compute_hirshfeld_charges` -- solving each free atom with the
SCF settings of the `result` it is given. `proatom="slater"` selects the
single-exponential Slater proatom. The CLI-parity check (binding vs the CLI's
NPZ export on the same settings) lives in `test_validation_npz_export.py`.

Artifact hypothesis: if the default still used the Slater proatom, the default
and "slater" charges would be equal and the "ALL atoms fell back" warning would
be printed on the default path; both are asserted against. The warning check
has a positive control (the "slater" path must print it), so a capture that
sees nothing cannot pass it vacuously.

Run:
    OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest \\
        crates/ferric-python/tests/test_hirshfeld_proatom.py -q
    OPENBLAS_NUM_THREADS=1 uv run --no-sync pytest -m validation \\
        crates/ferric-python/tests/test_hirshfeld_proatom.py -q
"""

import json
from pathlib import Path

import numpy as np
import pytest

import ferric

ROOT = Path(__file__).resolve().parents[3]
WATER = ROOT / "testdata" / "molecules" / "water.xyz"
VAL_WATER = ROOT / "testdata" / "molecules" / "validation" / "h2o.xyz"
VAL_REF = (
    ROOT / "testdata" / "reference" / "validation" / "hirshfeld" / "h2o_cc-pvdz.json"
)
# Matches both the all-atoms and the some-atoms fallback warning.
FALLBACK_WARNING = "fell back to"

# Slater vs SCF proatom on water: 0.62 e at cc-pVDZ (validation reference);
# anything above 0.1 e shows the two paths are different proatoms.
SWAP_MIN = 0.1
# Full-chain bar of crates/ferric-rpa/tests/validation_hirshfeld.rs (ferric RHF
# + ferric free atoms vs the PySCF-based reference).
TOL_Q_CHAIN = 1e-6


@pytest.fixture(scope="module")
def water_sto3g():
    mol = ferric.Molecule.from_xyz(str(WATER), 0, 1)
    bs = ferric.BasisSet.bundled("sto-3g")
    rhf = ferric.run_rhf(mol, bs, max_iter=200, energy_conv=1e-10, density_conv=1e-9)
    assert rhf.converged
    return mol, bs, rhf


def test_default_is_scf_and_prints_no_fallback_warning(water_sto3g, capfd):
    mol, bs, rhf = water_sto3g
    q_default = ferric.hirshfeld_charges(mol, bs, rhf)
    err_default = capfd.readouterr().err
    q_scf = ferric.hirshfeld_charges(mol, bs, rhf, proatom="scf")
    capfd.readouterr()
    assert np.array_equal(q_default, q_scf)
    assert FALLBACK_WARNING not in err_default, err_default
    assert abs(sum(q_default)) < 1e-10


def test_slater_differs_and_warns(water_sto3g, capfd):
    mol, bs, rhf = water_sto3g
    q_scf = ferric.hirshfeld_charges(mol, bs, rhf)
    capfd.readouterr()
    q_slater = ferric.hirshfeld_charges(mol, bs, rhf, proatom="slater")
    err_slater = capfd.readouterr().err
    # Positive control for the warning capture above.
    assert FALLBACK_WARNING in err_slater, err_slater
    d = float(np.max(np.abs(np.asarray(q_scf) - np.asarray(q_slater))))
    print(f"scf vs slater proatom max|d| {d:.3f} e")
    assert d > SWAP_MIN


@pytest.mark.parametrize("bad", ["SCF", "Slater", "", "becke", "none"])
def test_unknown_proatom_raises(water_sto3g, bad):
    mol, bs, rhf = water_sto3g
    with pytest.raises(ValueError, match="proatom"):
        ferric.hirshfeld_charges(mol, bs, rhf, proatom=bad)


@pytest.mark.validation
def test_default_matches_reference_full_chain():
    """Binding default vs the PySCF-based reference, at the Rust row's
    full-chain bar (ferric RHF + ferric free atoms vs the reference's)."""
    ref = json.loads(VAL_REF.read_text())
    mol = ferric.Molecule.from_xyz(str(VAL_WATER), ref["charge"], ref["multiplicity"])
    bs = ferric.BasisSet.bundled("cc-pvdz")
    rhf = ferric.run_rhf(mol, bs, max_iter=400, energy_conv=1e-11, density_conv=1e-9)
    assert rhf.converged
    # State check first, so a different SCF solution fails as a state, not as
    # a charge (the Rust row asserts 1e-10 with its own SCF config).
    assert abs(rhf.energy - ref["scf"]["energy"]) < 1e-8
    for proatom, key in (("scf", "scf_proatom"), ("slater", "slater")):
        q = ferric.hirshfeld_charges(mol, bs, rhf, proatom=proatom)
        d = float(np.max(np.abs(np.asarray(q) - np.asarray(ref["charges"][key]))))
        print(
            f"proatom={proatom}: max|d| vs reference {d:.2e} e (tol {TOL_Q_CHAIN:.0e})"
        )
        assert d < TOL_Q_CHAIN
