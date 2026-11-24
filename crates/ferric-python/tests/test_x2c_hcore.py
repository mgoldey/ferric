"""AoIntegrals.hcore_x2c: the Rust spin-free X2C-1e core Hamiltonian.

* equal to the numpy prototype (scripts/prototypes/x2c_proto.py) to the
  rounding of the transformation;
* E_X2C - E_NR through run_rhf(fock_hook=) against PySCF sfx2c1e
  (testdata/reference/x2c_sfx2c1e_cc-pvdz.json, scripts/gen_pyscf_x2c_refs.py);
* refusals: ECP atoms, a non-positive c.

The hcore eigenvalue comparison with PySCF and the c -> infinity limit are in
crates/ferric-scf/tests/x2c.rs.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ferric = pytest.importorskip("ferric")
if not hasattr(ferric.AoIntegrals, "hcore_x2c"):
    pytest.skip(
        "loaded ferric build has no AoIntegrals.hcore_x2c", allow_module_level=True
    )

ROOT = Path(__file__).resolve().parents[3]
REF = json.loads((ROOT / "testdata/reference/x2c_sfx2c1e_cc-pvdz.json").read_text())
LIGHT_SPEED = REF["light_speed"]
# SCF energy_conv of each of the four runs (two ferric, two PySCF); the bar on
# E_X2C - E_NR is their sum, as in the prototype.
CONV = REF["conv_tol"]

_spec = importlib.util.spec_from_file_location(
    "x2c_proto", ROOT / "scripts/prototypes/x2c_proto.py"
)
x2c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(x2c)


@pytest.fixture(scope="module")
def systems(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("x2c")
    return {label: x2c.System(label, tmp=tmp) for label in ("Ne", "Ar", "Kr", "HBr")}


def _bs():
    return ferric.BasisSet.bundled("cc-pvdz")


@pytest.mark.parametrize("label", ["Ne", "Kr"])
def test_equals_the_prototype_matrix(systems, label):
    """Both codes evaluate the same formula from the same integrals with
    different LAPACK paths; each carries the transformation's rounding,
    n_unc eps cond(S_u) max|T_u + V_u| (S_u^(+-1/2) applied to matrices of
    that size), so the bar is twice that."""
    sys_ = systems[label]
    h_rust = ferric.AoIntegrals(sys_.mol, _bs()).hcore_x2c()
    h_proto = sys_.ints.kinetic() + sys_.ints.nuclear() + sys_.delta_h(LIGHT_SPEED)
    es = np.linalg.eigvalsh(sys_.s_u)
    n = sys_.s_u.shape[0]
    bar = (
        2
        * n
        * np.finfo(float).eps
        * (es[-1] / es[0])
        * np.abs(sys_.t_u + sys_.v_u).max()
    )
    dev = np.abs(h_rust - h_proto).max()
    print(f"{label}: max|h_rust - h_proto| {dev:.2e} (bar {bar:.2e})")
    assert dev <= bar


def test_default_c_is_pyscf_light_speed():
    mol = ferric.Molecule.from_xyz_string("1\n\nNe 0 0 0\n")
    ints = ferric.AoIntegrals(mol, _bs())
    assert np.array_equal(ints.hcore_x2c(), ints.hcore_x2c(c=LIGHT_SPEED))


@pytest.mark.parametrize("label", ["Ne", "Ar", "Kr", "HBr"])
def test_energy_shift_matches_pyscf(systems, label):
    sys_ = systems[label]
    ints = ferric.AoIntegrals(sys_.mol, _bs())
    dh = ints.hcore_x2c() - ints.hcore()

    def hook(d):
        # Constant one-electron term: E_hook = Tr[D dH] makes the total the
        # energy of the core Hamiltonian hcore + dH (x2c_derivation.md).
        return float(np.sum(d * dh)), dh

    kw = dict(energy_conv=CONV, density_conv=1e-8, max_iter=200)
    nr = ferric.run_rhf(sys_.mol, _bs(), **kw)
    rel = ferric.run_rhf(sys_.mol, _bs(), fock_hook=hook, **kw)
    assert nr.converged and rel.converged
    ref = REF["systems"][label]
    d_f = rel.energy - nr.energy
    d_p = ref["e_x2c"] - ref["e_nr"]
    print(f"{label}: dE ferric {d_f:.10f} PySCF {d_p:.10f} |ddE| {abs(d_f - d_p):.2e}")
    assert abs(d_f - d_p) <= 4 * CONV


def test_ecp_atom_raises_value_error():
    mol = ferric.Molecule.from_xyz_string("1\n\nXe 0 0 0\n")
    ints = ferric.AoIntegrals(mol, ferric.BasisSet.bundled("cc-pvdz-pp"))
    with pytest.raises(ValueError, match="ECP"):
        ints.hcore_x2c()


@pytest.mark.parametrize("c", [0.0, -1.0, float("nan")])
def test_nonpositive_c_raises_value_error(c):
    mol = ferric.Molecule.from_xyz_string("1\n\nNe 0 0 0\n")
    with pytest.raises(ValueError, match="speed of light"):
        ferric.AoIntegrals(mol, _bs()).hcore_x2c(c=c)
