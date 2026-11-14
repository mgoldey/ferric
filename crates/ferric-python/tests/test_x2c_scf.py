"""relativistic="x2c1e" on the Python SCF drivers.

* The driver route equals the block route: run_rhf / run_uhf / run_rohf with
  relativistic="x2c1e" against the same SCF with the constant fock_hook
  V = AoIntegrals.hcore_x2c() - hcore(), E_hook = Tr[D V]. Each energy is
  within energy_conv of its stationary value, so the bar is 2 energy_conv.
* run_dft / run_uks / run_roks thread the kwarg (the X2C shift appears and the
  certificate records it; the Kohn-Sham driver-vs-hook anchor is in
  crates/ferric-scf/tests/x2c_scf.rs).
* Refusals: an unknown spelling, with_gradient=True, run_optimize and
  run_frequencies (scf_relativistic=, refused before any SCF), an ECP atom.
"""

from __future__ import annotations

import numpy as np
import pytest

ferric = pytest.importorskip("ferric")
if not hasattr(ferric.AoIntegrals, "hcore_x2c"):
    pytest.skip(
        "loaded ferric build has no AoIntegrals.hcore_x2c", allow_module_level=True
    )

CONV = 1e-10
KW = dict(energy_conv=CONV, density_conv=1e-8)


def _hbr(charge=0, mult=1):
    return ferric.Molecule.from_xyz_string(
        "2\n\nH 0 0 0\nBr 0 0 1.414\n", charge=charge, multiplicity=mult
    )


def _bs():
    return ferric.BasisSet.bundled("cc-pvdz")


def _shift(mol):
    ints = ferric.AoIntegrals(mol, _bs())
    return ints.hcore_x2c() - ints.hcore()


def _assert_same(label, driver, block):
    d = abs(driver.energy - block.energy)
    print(f"{label}: driver {driver.energy:.12f} block {block.energy:.12f} |d| {d:.2e}")
    assert d <= 2 * CONV
    names = [a["name"] for a in driver.certificate["approximations"]]
    assert "x2c1e" in names
    assert driver.certificate["grade"] == "ungraded"


def test_run_rhf_matches_the_fock_hook_block_route():
    mol = _hbr()
    dh = _shift(mol)
    driver = ferric.run_rhf(mol, _bs(), relativistic="x2c1e", **KW)
    block = ferric.run_rhf(
        mol, _bs(), fock_hook=lambda d: (float(np.sum(d * dh)), dh), **KW
    )
    _assert_same("run_rhf", driver, block)
    nr = ferric.run_rhf(mol, _bs(), **KW)
    assert abs(driver.energy - nr.energy) > 1.0
    assert "x2c1e" not in [a["name"] for a in nr.certificate["approximations"]]


@pytest.mark.parametrize("driver_fn", ["run_uhf", "run_rohf"])
def test_open_shell_matches_the_fock_hook_block_route(driver_fn):
    mol = _hbr(1, 2)
    dh = _shift(mol)
    run = getattr(ferric, driver_fn)

    def hook(da, db):
        return float(np.sum((da + db) * dh)), dh, dh

    driver = run(mol, _bs(), relativistic="x2c1e", **KW)
    block = run(mol, _bs(), fock_hook=hook, **KW)
    _assert_same(driver_fn, driver, block)


@pytest.mark.parametrize(
    "call",
    [
        lambda rel: ferric.run_dft(_hbr(), _bs(), functional="PBE", relativistic=rel),
        lambda rel: ferric.run_ksdft(_hbr(), _bs(), functional="PBE", relativistic=rel),
        lambda rel: ferric.run_uks(_hbr(1, 2), _bs(), "PBE", relativistic=rel),
        lambda rel: ferric.run_roks(_hbr(1, 2), _bs(), "PBE", relativistic=rel),
    ],
    ids=["run_dft", "run_ksdft", "run_uks", "run_roks"],
)
def test_kohn_sham_drivers_thread_the_kwarg(call):
    nr, rel = call(None), call("x2c1e")
    # run_dft returns a DftResult (total_energy); the others carry energy.
    nr_e, rel_e = (getattr(r, "energy", None) or r.total_energy for r in (nr, rel))
    # The X2C shift of HBr is about -26 Ha; 1 Ha is far below it and far
    # above any grid or fitting difference between the two runs.
    assert rel_e - nr_e < -1.0
    assert "x2c1e" in [a["name"] for a in rel.certificate["approximations"]]


@pytest.mark.parametrize("bad", ["X2C1E", "x2c", "sfx2c1e"])
def test_unknown_spelling_raises_value_error(bad):
    with pytest.raises(ValueError, match="relativistic"):
        ferric.run_rhf(_hbr(), _bs(), relativistic=bad)


def test_with_gradient_raises():
    with pytest.raises(ValueError, match='relativistic = "x2c1e" has no'):
        ferric.run_rhf(_hbr(), _bs(), relativistic="x2c1e", with_gradient=True)
    with pytest.raises(ValueError, match='relativistic = "x2c1e" has no'):
        ferric.run_uks(
            _hbr(1, 2), _bs(), "PBE", relativistic="x2c1e", with_gradient=True
        )


def test_geometry_tasks_raise_before_any_scf():
    with pytest.raises(ValueError, match='relativistic = "x2c1e" has no'):
        ferric.run_optimize(_hbr(), "cc-pvdz", scf_relativistic="x2c1e")
    with pytest.raises(ValueError, match='relativistic = "x2c1e" has no'):
        ferric.run_frequencies(_hbr(), "cc-pvdz", scf_relativistic="x2c1e")


def test_ecp_atom_raises():
    mol = ferric.Molecule.from_xyz_string("2\n\nH 0 0 0\nI 0 0 1.61\n")
    with pytest.raises(RuntimeError, match="ECP"):
        ferric.run_rhf(mol, ferric.BasisSet.bundled("def2-svp"), relativistic="x2c1e")
